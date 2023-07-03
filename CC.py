import math
from random import *
import numpy as np
import pandas as pd
from data_process.ProcessedData import ProcessedData

class CC(ProcessedData):

    def __init__(self, raw_data):
        super().__init__(raw_data)
        self.rest_columns = raw_data.rest_columns

    def process(self):
        if len(self.label_df) > 1:
            equal_zero_index = (self.label_df != 1).values
            equal_one_index = ~equal_zero_index


            fail_feature = np.array(self.feature_df[equal_one_index])

            ex_index=[]
            for temp in fail_feature:
                for i in range(len(temp)):
                    if temp[i] == 0:
                        ex_index.append(i)
            select_index=[]
            for i in range(len(self.feature_df.values[0])):
                if i not in ex_index:
                    select_index.append(i)

            select_index=list(set(select_index))
            sel_feature = self.feature_df.values.T[select_index].T
            columns = self.feature_df.columns[select_index]
            self.feature_df = pd.DataFrame(sel_feature, columns=columns)
            #print(self.feature_df.shape)


            equal_zero_index = (self.label_df != 1).values
            equal_one_index = ~equal_zero_index

            pass_feature = np.array(self.feature_df[equal_zero_index])
            fail_feature = np.array(self.feature_df[equal_one_index])

            centr_cc=np.ones(fail_feature.shape[1])
            dist = []
            for i in range(len(pass_feature)):
                dist.append(np.sqrt(np.sum(np.square(pass_feature[i] - centr_cc))))
            max_dis = max(dist)
            index = dist.index(max_dis)

            centr_tp = pass_feature[index]
            #print(pass_feature)
            ccc=centr_tp
            ctp=centr_cc
            Tcc = []
            Ttp = []
            while ~((ccc == centr_cc).all()) and ~((ctp == centr_tp).all()):
                Tcc = []
                Ttp = []
                for i in range(len(pass_feature)):
                    distcc = np.sqrt(np.sum(np.square(pass_feature[i] - centr_cc)))
                    disttp = np.sqrt(np.sum(np.square(pass_feature[i] - centr_tp)))
                    if disttp > distcc:
                        Tcc.append(pass_feature[i])
                    else:
                        Ttp.append(pass_feature[i])
                ccc=centr_cc
                ctp=centr_tp
                centr_cc=np.average(Tcc, axis=0)
                centr_tp=np.average(Ttp, axis=0)


            features_np = np.array(fail_feature)
            Tcc1 = np.array(Tcc)
            Ttp1 = np.array(Ttp)
            #print(Tcc1.shape)
            #print(Ttp1.shape)
            compose_tmp = np.vstack((features_np, Tcc1))
            compose_feature = np.vstack((compose_tmp, Ttp1))

            fnum=len(compose_tmp)
            pnum=len(Ttp)
            flabel = np.ones(fnum).reshape((-1, 1))
            plabel = np.zeros(pnum).reshape((-1, 1))
            compose_label = np.vstack((flabel, plabel))

            self.label_df = pd.DataFrame(compose_label, columns=['error'], dtype=float)
            self.feature_df = pd.DataFrame(compose_feature, columns=self.feature_df.columns, dtype=float)

            #self.data_df = pd.concat([self.feature_df, self.label_df], axis=1)
            #print(self.feature_df.shape)


