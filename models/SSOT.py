import numpy as np
import ot
from scipy.spatial.distance import cdist
from utils.metrics import val_model_binary
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model

class DeepSemiOT(object):
    def __init__(self, model, batch_size, n_class, optim, label_percent, sloss=1.0, tloss=1.0, gloss=1.0, closs=1.0, int_lr=0.01, ot_method='sinkhorn',
                 ot_alpha=1, lr_decay=True, verbose=1):
                     
        self.model = model   # target model
        self.batch_size = batch_size
        self.n_class= n_class
        self.optimizer= optim
        self.gamma = K.zeros(shape=(self.batch_size, self.batch_size)) # initialize the gamma (coupling in OT) with zeros
        self.sloss = K.variable(sloss) # weight for source classification
        self.tloss = K.variable(tloss) # weight for target classification
        self.gloss = K.variable(gloss)
        self.closs = K.variable(closs)

        self.verbose = verbose
        self.int_lr =int_lr  # initial learning rate
        self.lr_decay= lr_decay
        self.label_percent = label_percent

        self.ot_method = ot_method
        self.ot_alpha= ot_alpha  # weight for the alpha term
        self.jdot_alpha = 0.01

        self.ys = np.zeros([batch_size, 2])
        self.yt = np.zeros([int(batch_size * label_percent), 2])

        # some variables for class center loss, 128 is the feature's dimension
        self.cc_s0 = K.zeros([128])
        self.cc_s1 = K.zeros([128])
        self.cc_t0 = K.zeros([128])
        self.cc_t1 = K.zeros([128])

        self.iter = 0 # record current iteration
        
        # target classification cross ent loss and source cross entropy
        def classifier_cat_loss(y_true, y_pred):
            '''
            classifier loss based on categorical cross entropy in the target domain
            0: batch_size - is source samples
            batch_size: batch_size+int(batch_size*label_percent) - is target labeled samples
            batch_size+int(batch_size*label_percent): end - is target unlabeled samples
            self.gamma - is the optimal transport plan
            '''
            ys = y_true[:batch_size,:] # source true labels
            yt = y_true[batch_size:batch_size+int(batch_size*label_percent),:] # target true labels

            source_ypred = y_pred[:batch_size, :]  # source prediction
            ypred_t = y_pred[batch_size:batch_size+int(batch_size*label_percent),:] # target labeled prediction
            ypred_tt = y_pred[batch_size+int(batch_size*label_percent):, :] # target unlabeled prediction

            source_loss = K.mean(K.categorical_crossentropy(ys, source_ypred)) # source cross entropy
            target_loss = K.mean(K.categorical_crossentropy(yt, ypred_t)) # target labeled cross entropy

            # loss calculation based on double sum (sum_ij (ys^i, ypred_tt^j))
            ypred_tt = K.log(ypred_tt)
            loss = -K.dot(ys, K.transpose(ypred_tt))
            group_loss = K.sum(self.gamma[:, len(yt):] * loss)

            # print loss value
            # print(K.print_tensor(group_loss, message='group_loss = '))
            # print(K.print_tensor(source_loss, message='source_loss = '))
            # print(K.print_tensor(target_loss, message='target_loss = '))

            # returns source loss + target loss + group loss
            return self.tloss * target_loss + self.gloss * group_loss + self.sloss * source_loss

        self.classifier_cat_loss = classifier_cat_loss
        
        # L2 distance
        def L2_dist(x,y):
            '''
            compute the squared L2 distance between two matrics
            '''
            dist = K.reshape(K.sum(K.square(x),1), (-1,1)) # n*1
            dist += K.reshape(K.sum(K.square(y),1), (1,-1)) # n * 1 + 1 * n = n * n
            dist -= 2.0*K.dot(x, K.transpose(y))  
            return dist

        def L2_dist_center(x, y):
            '''
            compute the squared L2 distance between two vectors
            '''
            dist = K.sum(K.square(x))
            dist += K.sum(K.square(y))
            dist -= 2.0 * K.sum(x * y)
            return dist

       # feature allignment loss
        def align_loss(y_true, y_pred):
            '''
            source and target alignment loss in the intermediate layers of the target model
            allignment is performed in the target model (both source and target features are from target model)
            y-true - is dummy value( that is full of zeros)
            y-pred - is the value of intermediate layers in the target model
            1:batch_size - is source samples
            batch_size:end - is target samples            
            '''
            gs = y_pred[:batch_size,:] # source domain features
            gt = y_pred[batch_size:,:] # target domain features
            gdist = L2_dist(gs,gt)

            align_loss = K.sum(self.gamma * (gdist))

            cc_loss_s0 = K.sum(K.square(K.dot(K.transpose(gs-self.cc_s0),K.reshape(self.ys[:,0], (-1, 1)))))/K.sum(self.ys[:,0])
            cc_loss_s1 = K.sum(K.square(K.dot(K.transpose(gs-self.cc_s1),K.reshape(self.ys[:,1], (-1, 1)))))/K.sum(self.ys[:,1])
            # print("cc_loss_s0, s1: ", cc_loss_s0, cc_loss_s1)

            gt_left = gt[ :len(self.yt), :]

            cc_loss_t0 = K.sum(K.square(K.dot(K.transpose(gt_left-self.cc_t0),K.reshape(self.yt[:,0], (-1, 1)))))/K.sum(self.yt[:,0])
            cc_loss_t1 = K.sum(K.square(K.dot(K.transpose(gt_left-self.cc_t1),K.reshape(self.yt[:,1], (-1, 1)))))/K.sum(self.yt[:,1])

            cc_dis_s = L2_dist_center(self.cc_s0, self.cc_s1)
            cc_dis_t = L2_dist_center(self.cc_t0, self.cc_t1)
            cc_loss =  cc_loss_s0 + cc_loss_s1 + cc_loss_t0 + cc_loss_t1 - (cc_dis_s + cc_dis_t) # loss of cc

            # print loss
            # print(K.print_tensor(cc_dis_s, message='cc_dis_s = ')) #cc_dis_s.eval(), cc_dis_t.eval())
            # print(K.print_tensor(cc_dis_t, message='cc_dis_t = '))
            # print(K.print_tensor(cc_loss_s0, message='cc_loss_s0 = '))
            # print(K.print_tensor(cc_loss_s1, message='cc_loss_s1 = '))
            # print(K.print_tensor(cc_loss_t0, message='cc_loss_t0 = '))
            # print(K.print_tensor(cc_loss_t1, message='cc_loss_t1 = '))
            # print(K.print_tensor(align_loss, message='align_loss = '))

            return  self.ot_alpha * ( align_loss ) + cc_loss * self.closs

        self.align_loss = align_loss
        
        def feature_extraction(model, data, out_layer_num=-2):
            '''
            extract the features from the pre-trained model
            inp_layer_num - input layer
            out_layer_num -- from which layer to extract the features
            '''
            intermediate_layer_model = Model(inputs=model.layers[1].layers[1].input,
                             outputs=model.layers[1].layers[out_layer_num].output)
            intermediate_output = intermediate_layer_model.predict(data)
            return intermediate_output
        self.feature_extraction = feature_extraction

 
    def fit(self, source_traindata, ys_label, target_traindata1, yt_label, target_traindata2, Xtest, ytest_cat, target_label = None,
            n_iter=5000, cal_bal=True, sample_size=None):
        '''
        source_traindata - source domain training data
        ys_label - source data true labels
        target_traindata1 - target domain training data with labels
        yt_label - target data true labels
        target_traindata2 - target domain training data without labels
        cal_bal - True: source domain samples are equally represented from
                        all the classes in the mini-batch (that is, n samples from each class)
                - False: source domain samples are randomly sampled
        target_label - is not None  : compute the target auc over the iterations
        '''

        ns = source_traindata.shape[0]
        nt_l = target_traindata1.shape[0]
        nt_u = target_traindata2.shape[0]

        method = self.ot_method # for optimal transport
        fe_size = self.model.output_shape[1][1]
        t_loss =[]
        
        # function to sample n samples from each class
        def mini_batch_class_balanced(label, sample_size=100, shuffle=True, domain='source'):
            ''' sample the mini-batch with class balanced
            '''
            label = np.argmax(label, axis=1)

            if shuffle:
                rindex = np.random.permutation(len(label))
                label = label[rindex]

            n_class = len(np.unique(label))
            index = []
            for i in range(n_class):
                s_index = np.nonzero(label == i)
                s_ind = np.random.permutation(s_index[0])
                if i==1:
                    index = np.append(index, s_ind[0: sample_size])
                elif i == 0 and domain =='source':
                    index = np.append(index, s_ind[0: self.batch_size-sample_size])
                elif i == 0 and domain == 'target':
                    index = np.append(index, s_ind[0: int(self.label_percent * self.batch_size - sample_size)])
            index = np.array(index, dtype=int)
            return index
            
         # target model compliation and optimizer
        self.model.compile(optimizer= self.optimizer, loss =[self.classifier_cat_loss, self.align_loss])
        # set the learning rate
        K.set_value(self.model.optimizer.lr, self.int_lr) 
        
        for i in range(n_iter):
            self.iter = i

            if i % 100 == 0:
                # randomly select 50% samples to calculate the class centers
                rd_idx_s = np.random.choice(ns, int(0.5*ns))
                rd_source_feas = self.model.predict(source_traindata[rd_idx_s])[1]

                rd_idx_t = np.random.choice(nt_l, int(0.5*nt_l))
                rd_target_feas = self.model.predict(target_traindata1[rd_idx_t])[1]
                # print("target_feas: ",  target_feas.shape)

                cc_s = np.dot(np.transpose(rd_source_feas), ys_label[rd_idx_s]) / (ys_label[rd_idx_s].sum(axis=0))
                K.set_value(self.cc_s0, cc_s[:, 0])
                K.set_value(self.cc_s1, cc_s[:, 1])

                cc_t = np.dot(np.transpose(rd_target_feas), yt_label[rd_idx_t]) / (yt_label[rd_idx_t].sum(axis=0))
                K.set_value(self.cc_t0, cc_t[:, 0])
                K.set_value(self.cc_t1, cc_t[:, 1])

                # cc_dis_s = np.mean(np.sum(np.square(self.cc_s0 - self.cc_s1)))
                # cc_dis_t = np.mean(np.sum(np.square(self.cc_t0 - self.cc_t1)))
                # print(f"Iter: {i}, the class center distance of source {cc_dis_s} and target {cc_dis_t}")

            if self.lr_decay and i > 0 and i % 5000 ==0:
                lr = K.get_value(self.model.optimizer.lr)
                K.set_value(self.model.optimizer.lr, lr*0.1)
             
            # source domain mini-batch indexes
            if cal_bal:
                s_ind = mini_batch_class_balanced(ys_label, sample_size=sample_size, domain='source')
                self.sbatch_size = len(s_ind)
                t_ind_l = mini_batch_class_balanced(yt_label, sample_size=int(sample_size * self.label_percent), domain='target')
                self.tbatch_size_l = len(t_ind_l)
            else:
                s_ind = np.random.choice(ns, self.batch_size)
                self.sbatch_size = self.batch_size
                # target domain mini-batch indexes
                t_ind_l = np.random.choice(nt_l, int(self.batch_size * self.label_percent))
                self.tbatch_size_l = len(t_ind_l)
            t_ind_u = np.random.choice(nt_u, self.sbatch_size - self.tbatch_size_l)

            # source and target domain mini-batch samples 
            xs_batch, ys = source_traindata[s_ind], ys_label[s_ind]
            xt_batch, yt = target_traindata1[t_ind_l], yt_label[t_ind_l]
            xtt_batch = target_traindata2[t_ind_u]

            self.ys = ys
            self.yt = yt

             # dummy target outputs for the keras model
            l_dummy = -1 * np.ones([xtt_batch.shape[0], 2])  # for target samples
            # for intermediate layer feature values in the target model
            g_dummy = np.zeros((2*self.sbatch_size, fe_size))
            
            # concat of source and target samples and prediction
            modelpred = self.model.predict(np.vstack((xs_batch, xt_batch, xtt_batch)))
           
            # modelpred[0] - is softmax prob, and modelpred[1] - is intermediate layer
            gs_batch = modelpred[1][:self.sbatch_size, :]
            gt_batch = modelpred[1][self.sbatch_size:(self.sbatch_size+self.tbatch_size_l), :]
            gtt_batch = modelpred[1][(self.sbatch_size+self.tbatch_size_l):, :]
            gt_all_batch = modelpred[1][self.sbatch_size:, :]

            # softmax prediction of target samples
            # fs_pred = modelpred[0][:self.sbatch_size,:]
            ft_pred = modelpred[0][self.sbatch_size:(self.sbatch_size+self.tbatch_size_l), :]
            ftt_pred = modelpred[0][(self.sbatch_size + self.tbatch_size_l):, :]
            ft_all_pred= modelpred[0][self.sbatch_size:]

            # ground metric for the target classification loss
            C0_1 = cdist(gt_batch, gs_batch, metric='sqeuclidean')
            C0_2 = cdist(gtt_batch, gs_batch, metric='sqeuclidean')

            # only positive samples are matched, the cost is 0
            ys_list = np.argmax(ys, 1)
            yt_list = np.argmax(yt, 1)
            R = np.dot(yt_list.reshape(len(yt_list), 1), ys_list.reshape(1, len(ys_list)))
            R = np.ones_like(R) - R

            C1 = cdist(ftt_pred, ys, metric='sqeuclidean')

            # OT ground metric
            C = K.transpose(np.vstack((R*C0_1, C1*C0_2)))
                             
            # OT optimal coupling (gamma)
            if method == 'emd':
                # print(gs_batch.shape, gt_batch.shape, gtt_batch.shape, C.shape)
                gamma=ot.emd(ot.unif(gs_batch.shape[0]),ot.unif(gt_batch.shape[0]+gtt_batch.shape[0]),C)

            elif method == 'sinkhorn':
                gamma =ot.sinkhorn(ot.unif(gs_batch.shape[0]),ot.unif(gt_batch.shape[0]+gtt_batch.shape[0]), C, reg=0.1)

            # update the computed gamma
            if not np.isnan(gamma).any():
                K.set_value(self.gamma, gamma)
            else:
                print('Something wrong in OT!')
                break

            # train the keras model on batch
            data = np.vstack((xs_batch, xt_batch, xtt_batch))
            hist= self.model.train_on_batch([data], [np.vstack((ys, yt, l_dummy)), g_dummy])
            
            t_loss.append(hist[0])

            if self.verbose:
                if (i+1)%1000==0:
                    print ('Iter: {}, tl_loss = {:f}, fe_loss = {:f},  tot_loss = {:f}'.format(i+1, hist[1], hist[2], hist[0]))
                    if target_label is not None:
                        probs_t = self.model.predict(Xtest)[0][:,1]
                        val_model_binary(ytest_cat[:, 1], probs_t)
        return self

    def predict(self, data):
        '''
        return the predict probability
        '''
        ypred = self.model.predict(data)
        return ypred

    def evaluate(self, data):
        '''
        return the predict probability of class 1 (with Sepsis in the setting)
        '''
        ypred = self.model.predict(data)
        probs = ypred[0][:,1]
        return probs
