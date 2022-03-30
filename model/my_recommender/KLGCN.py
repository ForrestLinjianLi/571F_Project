"""
Paper: LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
Author: Xiangnan He, Kuan Deng, Xiang Wang, Yan Li, Yongdong Zhang, and Meng Wang
Reference: https://github.com/hexiangnan/LightGCN
"""

import scipy.sparse as sp
import tensorflow as tf
import numpy as np
from model.AbstractRecommender import AbstractRecommender
from util import timer
from util import learner
from util import l2_loss, inner_product, log_loss
from util import get_info
from util import SumAggregator, ConcatAggregator, NeighborAggregator, TopAggregator, DiAggregator
from data import PairwiseSampler
from sklearn.metrics import f1_score, roc_auc_score
import time


class KLGCN(AbstractRecommender):
    def __init__(self, sess, dataset, config):
        super(KLGCN, self).__init__(dataset, config)
        self.lr = config['lr']
        self.reg = config['reg']
        self.emb_dim = config['embed_size']
        self.batch_size = config['batch_size']
        self.epochs = config["epochs"]
        self.n_layers = config['n_layers']
        self.n_iters = config['n_iters']
        self.aggregator = config['aggregator']
        self.n_neighbors = config['n_neighbors']
        self.loss_function = config['loss_function']
        self.loss_type = config['loss_type']
        self.concat_type = config['concat_type']
        self.c_alpha = config['concat_alpha']
        self.feature_transform = config['feature_transform']
        self.activation = config['activation']
        self.agg = config['agg']
        self.att_type = config['att_type']
        self.add_user = config['add_user']
        self.reverse = config['reverse']
        self.update_user = config['update_user']
        self.plus_user = config['plus_user']

        self.cur_epoch = 0
        self.dataset = dataset
        self.n_users, self.n_items = self.dataset.num_users, self.dataset.num_items
        self.user_pos_train = self.dataset.get_user_train_dict(by_time=False)
        self.user_pos_test = self.dataset.get_user_test_dict()
        self.all_users = list(self.user_pos_train.keys())
        test_users = list(self.user_pos_test.keys())
        test_user_num = config['test_user_num']
        if test_user_num and test_user_num < len(test_users):
            self.test_users = np.random.choice(test_users, size=test_user_num, replace=False)
        else:
            self.test_users = test_users

        self.norm_adj, self.adj_mat = self.create_adj_mat(config['adj_type'])
        self.sess = sess

        self.n_entities, self.n_relations, kg = self.dataset.get_kg(config, self.user_pos_train, add_user=self.add_user,
                                                                    reverse=self.reverse)
        self.adj_entity, self.adj_relation = self.dataset.get_adj(kg, self.n_neighbors, self.n_entities,
                                                                  add_user=self.add_user)
        self.constraint_mat = self.get_constraint_mat()
        if self.aggregator == 'sum':
            self.aggregator_class = SumAggregator
        elif self.aggregator == 'concat':
            self.aggregator_class = ConcatAggregator
        elif self.aggregator == 'neighbor':
            self.aggregator_class = NeighborAggregator
        elif self.aggregator == 'top':
            self.aggregator_class = TopAggregator
        elif self.aggregator == 'direct':
            self.aggregator_class = DiAggregator
        else:
            raise Exception("Unknown aggregator: " + self.aggregator)

    def get_constraint_mat(self):
        user_list, item_list = self.dataset.get_train_interactions()

        row = np.array(user_list, dtype=np.float32)
        col = np.array(item_list, dtype=np.float32)

        adj = sp.csr_matrix((np.ones_like(row, dtype=np.float32), (row, col)), shape=[self.n_users, self.n_items])
        user_deg = np.array(adj.sum(axis=1)).flatten()  # np.sum(adj, axis=1).reshape(-1)
        item_deg = np.array(adj.sum(axis=0)).flatten()  # np.sum(adj, axis=0).reshape(-1)

        beta_user_deg = (np.sqrt(user_deg + 1) / user_deg).reshape(-1, 1)
        beta_item_deg = (1 / np.sqrt(item_deg + 1)).reshape(1, -1)

        constraint_mat = beta_user_deg @ beta_item_deg  # n_user * m_item
        constraint_mat = np.array(constraint_mat, dtype=np.float32)

        return constraint_mat


    @timer
    def create_adj_mat(self, adj_type):
        user_list, item_list = self.dataset.get_train_interactions()
        user_np = np.array(user_list, dtype=np.int32)
        item_np = np.array(item_list, dtype=np.int32)
        ratings = np.ones_like(user_np, dtype=np.float32)
        n_nodes = self.n_users + self.n_items
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np+self.n_users)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj)
            print('generate single-normalized adjacency matrix.')
            return norm_adj.tocoo()

        if adj_type == 'plain':
            adj_matrix = adj_mat
            print('use the plain adjacency matrix')
        elif adj_type == 'norm':
            adj_matrix = normalized_adj_single(adj_mat + sp.eye(adj_mat.shape[0]))
            print('use the normalized adjacency matrix')
        elif adj_type == 'gcmc':
            adj_matrix = normalized_adj_single(adj_mat)
            print('use the gcmc adjacency matrix')
        elif adj_type == 'pre':
            # pre adjcency matrix
            rowsum = np.array(adj_mat.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj_tmp = d_mat_inv.dot(adj_mat)
            adj_matrix = norm_adj_tmp.dot(d_mat_inv)
            print('use the pre adjcency matrix')
        else:
            mean_adj = normalized_adj_single(adj_mat)
            adj_matrix = mean_adj + sp.eye(mean_adj.shape[0])
            print('use the mean adjacency matrix')

        return adj_matrix, adj_mat

    def _create_variable(self):

        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))
        self.neg_items = tf.placeholder(tf.int32, shape=(None,))
        self.labels = tf.placeholder(tf.float32, shape=(None,))

        self.weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()
        self.weights['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]), name='user_embedding')
        self.weights['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]), name='item_embedding')
        self.weights['entity_embedding'] = tf.Variable(initializer([self.n_entities - self.n_items, self.emb_dim]),
                                                       name='entity_embedding')  # except all items
        if self.reverse:
            self.weights['relation_embedding'] = tf.Variable(initializer([(self.n_relations + 1) * 2, self.emb_dim]),
                                                             name='relation_embedding')
        else:
            self.weights['relation_embedding'] = tf.Variable(initializer([self.n_relations, self.emb_dim]),
                                                             name='relation_embedding')


    def build_graph(self):
        self._create_variable()
        """
        LightGCN part
        """
        ua_embeddings, ia_embeddings = self._create_ultragcn_embedding()
        self.ia_embeddings = ia_embeddings
        # ua_embeddings, ia_embeddings = self._create_lightgcn_embed()
        """
        KGNN part
        """
        self.u_kg_embeddings, self.pos_i_kg_embeddings, self.neg_i_kg_embeddings, self.aggregators = self._create_kgcn_embed()
        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        """
        self.u_g_embeddings = tf.nn.embedding_lookup(ua_embeddings, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(ia_embeddings, self.pos_items)
        self.neg_i_g_embeddings = tf.nn.embedding_lookup(ia_embeddings, self.neg_items)

        if self.concat_type == 'stack':
            if self.plus_user:
                self.u_kg_embeddings = tf.expand_dims(self.u_kg_embeddings, axis=1)
                self.user_embeddings = tf.concat([self.u_g_embeddings, self.u_kg_embeddings], axis=1)
            else:
                self.user_embeddings = self.u_g_embeddings
            self.user_embeddings = tf.reduce_mean(self.user_embeddings, axis=1)

            self.pos_i_kg_embeddings = tf.expand_dims(self.pos_i_kg_embeddings, axis=1)
            self.pos_item_embeddings = tf.concat([self.pos_i_g_embeddings, self.pos_i_kg_embeddings], axis=1)
            self.pos_item_embeddings = tf.reduce_mean(self.pos_item_embeddings, axis=1)

            self.neg_i_kg_embeddings = tf.expand_dims(self.neg_i_kg_embeddings, axis=1)
            self.neg_item_embeddings = tf.concat([self.neg_i_g_embeddings, self.neg_i_kg_embeddings], axis=1)
            self.neg_item_embeddings = tf.reduce_mean(self.neg_item_embeddings, axis=1)
        else:
            self.u_g_embeddings = tf.reduce_mean(self.u_g_embeddings, axis=1)
            self.pos_i_g_embeddings = tf.reduce_mean(self.pos_i_g_embeddings, axis=1)
            self.neg_i_g_embeddings = tf.reduce_mean(self.neg_i_g_embeddings, axis=1)

            if self.concat_type == 'concat':
                if self.plus_user:
                    self.user_embeddings = tf.concat([self.u_kg_embeddings, self.u_g_embeddings], axis=1)
                else:
                    self.user_embeddings = self.u_g_embeddings

                self.pos_item_embeddings = tf.concat([self.pos_i_kg_embeddings, self.pos_i_g_embeddings], axis=1)
                self.neg_item_embeddings = tf.concat([self.neg_i_kg_embeddings, self.neg_i_g_embeddings], axis=1)

            elif self.concat_type == 'rate':
                if self.plus_user:
                    self.user_embeddings = self.c_alpha * self.u_kg_embeddings + (1 - self.c_alpha) * self.u_g_embeddings
                else:
                    self.user_embeddings = self.u_g_embeddings
                self.pos_item_embeddings = self.c_alpha * self.pos_i_kg_embeddings + (1 - self.c_alpha) * self.pos_i_g_embeddings
                self.neg_item_embeddings = self.c_alpha * self.neg_i_kg_embeddings + (1 - self.c_alpha) * self.neg_i_g_embeddings

            elif self.concat_type == 'add':
                if self.plus_user:
                    self.user_embeddings = self.u_kg_embeddings + self.u_g_embeddings
                else:
                    self.user_embeddings = self.u_g_embeddings
                self.pos_item_embeddings = self.pos_i_kg_embeddings + self.pos_i_g_embeddings
                self.neg_item_embeddings = self.neg_i_kg_embeddings + self.neg_i_g_embeddings

            else:
                 raise ValueError('Unknown concat type ' + self.concat_type)


        """
        *********************************************************
        Generate Predictions & Optimize via BPR loss.
        """
        self.mf_loss, self.ctr_loss, self.emb_loss = self.create_bpr_loss()

        if self.loss_type == 'mf&ctr':
            self.loss = self.mf_loss + self.ctr_loss + self.emb_loss
        elif self.loss_type == 'ctr':
            self.loss = self.ctr_loss + self.emb_loss
        elif self.loss_type == 'mf':
            self.loss = self.mf_loss + self.emb_loss
        elif self.loss_type == 'cross':
            if self.cur_epoch % 2 == 0:
                self.loss = self.ctr_loss + self.emb_loss
            else:
                self.loss = self.mf_loss + self.emb_loss
            self.cur_epoch += 1
        else:
            raise ValueError('Unknown loss type' + self.loss_type)

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _create_lightgcn_embed(self):
        adj_mat = self._convert_sp_mat_to_sp_tensor(self.norm_adj)

        ego_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding']], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):
            side_embeddings = tf.sparse_tensor_dense_matmul(adj_mat, ego_embeddings, name="sparse_dense")
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]

        all_embeddings = tf.stack(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)
        return u_g_embeddings, i_g_embeddings

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _create_kgcn_embed(self):
        # neg_items = self.neg_items[:, 0]
        if self.add_user:
            pos_i_indices, neg_i_indices = self.pos_items + self.n_users, self.neg_items + self.n_users
        else:
            pos_i_indices, neg_i_indices = self.pos_items, self.neg_items

        user_embeddings_pre = tf.nn.embedding_lookup(self.weights['user_embedding'], self.users)
        if self.add_user and self.update_user:
            user_embeddings = tf.nn.embedding_lookup(self.weights['user_embedding'], tf.tile(self.users, [3]))
            all_indices = tf.concat((self.users, pos_i_indices, neg_i_indices), axis=0)
        else:
            user_embeddings = tf.nn.embedding_lookup(self.weights['user_embedding'], tf.tile(self.users, [2]))
            all_indices = tf.concat((pos_i_indices, neg_i_indices), axis=0)

        batch_size = tf.shape(all_indices)[0]
        entities, relations = self.get_neighbors(all_indices, batch_size)
        embeddings, aggregators = self.aggregate(entities, relations, user_embeddings, batch_size)

        if self.add_user and self.update_user:
            embeddings = tf.reshape(embeddings, [3, -1, self.emb_dim])
            user_embeddings, pos_item_embeddings, neg_item_embeddings = embeddings[0], embeddings[1], embeddings[2]
        else:
            embeddings = tf.reshape(embeddings, [2, -1, self.emb_dim])
            user_embeddings, pos_item_embeddings, neg_item_embeddings = user_embeddings_pre, embeddings[0], embeddings[1]

        return user_embeddings, pos_item_embeddings, neg_item_embeddings, aggregators

    def _create_ultragcn_embedding(self):
        return tf.expand_dims(self.weights['user_embedding'], axis=1), tf.expand_dims(self.weights['item_embedding'], axis=1)

    def get_neighbors(self, seeds, batch_size):
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        relations = []
        for i in range(self.n_iters):
            neighbor_entities = tf.reshape(tf.gather(self.adj_entity, entities[i]), [batch_size, -1])
            neighbor_relations = tf.reshape(tf.gather(self.adj_relation, entities[i]), [batch_size, -1])
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations
    
    def aggregate(self, entities, relations, user_embeddings, batch_size):
        aggregators = []  # store all aggregators
        if self.add_user:
            entity_embeddings = tf.concat([self.weights['user_embedding'], self.weights['item_embedding'],
                                           self.weights['entity_embedding']], axis=0)
        else:
            entity_embeddings = tf.concat([self.weights['item_embedding'], self.weights['entity_embedding']], axis=0)

        entity_vectors = [tf.nn.embedding_lookup(entity_embeddings, i) for i in entities]
        relation_vectors = [tf.nn.embedding_lookup(self.weights['relation_embedding'], i) for i in relations]

        for i in range(self.n_iters):
            if i == self.n_iters - 1:
                aggregator = self.aggregator_class(self.n_neighbors, batch_size, self.emb_dim, act=tf.nn.tanh,
                                                   feature_transform=self.feature_transform, activation=self.activation,
                                                   att_type=self.att_type, agg=self.agg)
            else:
                aggregator = self.aggregator_class(self.n_neighbors, batch_size, self.emb_dim,
                                                   feature_transform=self.feature_transform, activation=self.activation,
                                                   att_type=self.att_type, agg=self.agg)
            aggregators.append(aggregator)

            entity_vectors_next_iter = []
            for hop in range(self.n_iters - i):
                shape = [batch_size, -1, self.n_neighbors, self.emb_dim]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vectors=tf.reshape(entity_vectors[hop + 1], shape),
                                    neighbor_relations=tf.reshape(relation_vectors[hop], shape),
                                    user_embeddings=user_embeddings,
                                    masks=None)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter
        res = tf.reshape(entity_vectors[0], [batch_size, self.emb_dim])
        return res, aggregators

    def create_bpr_loss(self):
        batch_neg_item_indices = tf.random.uniform(shape=[tf.shape(self.users)[0], 800], maxval=self.n_items, dtype=tf.int64)

        batch_user_weights = tf.gather(self.constraint_mat, self.users)
        pos_weights = tf.gather(batch_user_weights, self.pos_items, batch_dims=-1)
        neg_weights = tf.gather(batch_user_weights, batch_neg_item_indices, batch_dims=-1)

        embedded_neg_items = tf.squeeze(tf.nn.embedding_lookup(self.ia_embeddings, batch_neg_item_indices), axis=2)
        pos_logits = tf.reduce_sum(self.u_g_embeddings * self.pos_i_g_embeddings, axis=-1)
        neg_logits = tf.reduce_sum((tf.expand_dims(self.u_g_embeddings, axis=1) * embedded_neg_items), axis=-1)

        pos_losses = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=pos_logits,
            labels=tf.ones_like(pos_logits)
        )
        neg_losses = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=neg_logits,
            labels=tf.zeros_like(neg_logits)
        )

        # mf_loss = pos_logits_u * (1e-6 + pos_weights) + tf.reduce_mean(neg_logits_u * (1e-6 + neg_weights), axis=1) * 300
        mf_loss = pos_losses * (1e-6 + pos_weights) + tf.reduce_mean(neg_losses * (1e-6 + neg_weights), axis=1) * 300 + tf.reduce_sum(log_loss(self.pos_scores - self.neg_scores))

        self.pos_scores = inner_product(self.user_embeddings, self.pos_item_embeddings)
        self.neg_scores = inner_product(self.user_embeddings, self.neg_item_embeddings)
        self.pos_score_normalized = tf.sigmoid(self.pos_scores)
        self.neg_score_normalized = tf.sigmoid(self.neg_scores)
        scores = tf.concat([self.pos_scores, self.neg_scores], axis=0)
        if self.feature_transform:
            for aggregator in self.aggregators:
                regularizer = regularizer + l2_loss(aggregator.weights)

        regularizer = l2_loss(self.weights['user_embedding'], self.weights['item_embedding'],
                              self.weights['entity_embedding'], self.weights['relation_embedding'])

        ctr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=scores))
        emb_loss = self.reg * regularizer

        return tf.reduce_sum(mf_loss), ctr_loss, emb_loss

    # def create_bpr_loss(self):
    #     self.pos_scores = inner_product(self.user_embeddings, self.pos_item_embeddings)
    #     self.neg_scores = inner_product(self.user_embeddings, self.neg_item_embeddings)
    #     scores = tf.concat([self.pos_scores, self.neg_scores], axis=0)
    #
    #     regularizer = l2_loss(self.weights['user_embedding'], self.weights['item_embedding'],
    #                           self.weights['entity_embedding'], self.weights['relation_embedding'])
    #     self.pos_score_normalized = tf.sigmoid(self.pos_scores)
    #     self.neg_score_normalized = tf.sigmoid(self.neg_scores)
    #     if self.feature_transform:
    #         for aggregator in self.aggregators:
    #             regularizer = regularizer + l2_loss(aggregator.weights)
    #
    #     mf_loss = tf.reduce_sum(log_loss(self.pos_scores - self.neg_scores))
    #     ctr_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=scores))
    #
    #     emb_loss = self.reg * regularizer
    #
    #     return mf_loss, ctr_loss, emb_loss

    def train_model(self):
    
        data_iter = PairwiseSampler(self.dataset, neg_num=1, batch_size=self.batch_size, shuffle=True, user_pos_dict=self.user_pos_train)
        test_iter = PairwiseSampler(self.dataset, neg_num=1, batch_size=self.batch_size, shuffle=True, user_pos_dict=self.user_pos_test)
        best_result, cur_epoch, cur_auc, cur_f1 = None, None, None, None
        step = 0
        for epoch in range(self.epochs):
            for bat_users, bat_pos_items, bat_neg_items in data_iter:
                feed = {self.users: bat_users,
                          self.pos_items: bat_pos_items,
                          self.neg_items: bat_neg_items,
                          self.labels: len(bat_pos_items) * [1] + len(bat_neg_items) * [0]}
                self.sess.run([self.opt], feed_dict=feed)

            auc_list, f1_list = [], []
            for bat_users, bat_pos_items, bat_neg_items in test_iter:
                feed = {self.users: bat_users,
                        self.pos_items: bat_pos_items,
                        self.neg_items: bat_neg_items}
                auc, f1 = self.ctr_eval(feed)
                auc_list.append(auc)
                f1_list.append(f1)
            final_auc, final_f1 = float(np.mean(auc_list)), float(np.mean(f1_list))
            result = self.evaluate_model(self.test_users)

            info = get_info(result, epoch, final_auc, final_f1)
            self.logger.info(info)

            if not best_result or result['Recall'][0] > best_result['Recall'][0]:
                best_result, cur_epoch, cur_auc, cur_f1 = result, epoch, final_auc, final_f1
                step = 0
            else:
                step += 1

            if self.dataset.dataset_name in ['movie', 'restaurant', 'yelp2018']:
                if step >= 10 or epoch == self.epochs - 1:
                    info = get_info(best_result, cur_epoch, cur_auc, cur_f1)
                    self.logger.info('-' * 27 + ' BEST RESULT ' + '-' * 27)
                    self.logger.info(info)
                    break
            else:
                if epoch == self.epochs - 1:
                    info = get_info(best_result, cur_epoch, cur_auc, cur_f1)
                    self.logger.info('-' * 27 + ' BEST RESULT ' + '-' * 27)
                    self.logger.info(info)
                    break

    # @timer
    def evaluate_model(self, users):
        return self.evaluator.evaluate(self, users)

    def predict(self, users, candidate_items=None):
        all_items = list(range(self.n_items))
        ratings = np.zeros((len(users), self.n_items))
        for idx, user in enumerate(users):
            for start in range(0, self.n_items, self.batch_size):
                items = all_items[start: start + self.batch_size]
                feed_dict = {self.users: [user] * len(items),
                             self.pos_items: items,
                             self.neg_items: items}
                             # self.neg_items: np.expand_dims(np.array(items), axis=1),}

                batch_ratings = self.sess.run(self.pos_scores, feed_dict=feed_dict)
                ratings[idx][start: start + self.batch_size] = batch_ratings
        if candidate_items is not None:
            ratings = [ratings[idx][u_item] for idx, u_item in enumerate(candidate_items)]
        return ratings

    def ctr_eval(self, feed_dict):
        pos_scores, neg_scores = self.sess.run([self.pos_score_normalized, self.neg_score_normalized], feed_dict)
        scores = np.concatenate((pos_scores, neg_scores), axis=0)
        assert len(pos_scores) == len(neg_scores)
        labels = np.array(len(pos_scores) * [1] + len(neg_scores) * [0])
        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        return auc, f1
