import pandas as pd
import json
import os
import numpy as np
from G2DB.core.hash import HashTable
from G2DB.core.attribute import Attribute
from BTrees.OOBTree import OOBTree

class Table:
    # info = {}
    # need a load() outside
    def __init__(self, attrls, info):
        self.data = {}
        self.datalist = []
        self.df = pd.DataFrame()
        self.name = info['name']
        self.attrls = attrls
        self.attrs = {} #{name: attributeobj}
        self.primary = info['primary']
        self.foreign = info['foreign']
        self.uniqueattr = {} # {attribute_name: {attibute_value: primarykey_value}}
        self.index={}   #{attr: idex_name}
        self.BTree={}   #{idex_name: BTree}
        self.flag = 0

        for attr in info['attrs']:
            temp = Attribute(attr)
            self.attrs[attr['name']] = temp
            if temp.unique:
                self.uniqueattr[attr['name']] = {}

        # TODO: moxiao add_index

    def add_index(self, attr, index_name, db, table):
        # no index on this attr before
        if not os.path.exists('index_info.npy'):
            info = []
            infoarr = np.array(info)
            np.save('index_info.npy', infoarr)
        else:
            # if index_info.npy existed, open it
            infoarr = np.load('index_info.npy', allow_pickle=True)
            info = infoarr.tolist()
            # check if the index was existed, index is the third attribute
            for item in iter(infoarr):
                if (item[2] == index_name): raise Exception(
                    '[ERROR]: Index  is existed.')

        info.append([table, attr, index_name])
        infoarr = np.array(info)
        print(info)
        np.save('index_info.npy', info)
        # create index
        dict0 = {}
        dict1 = {}
        dict2 = {}
        dict3 = {}
        dict4 = {}
        dict5 = {}
        dict6 = {}
        hashlist = [dict0, dict1, dict2, dict3, dict4, dict5, dict6]
        df = db.tables[table].search(['*'], [], [], [], False)
        length = df.shape[0]
        for i in range(length):
            row = df.iloc[i]
            value = int(row[attr])
            # print(value)
            hashtable = HashTable(7)
            hash_index = hashtable.hash(value)
            # print(hash_index)
            if hash_index == 0:
                dict0[value] = row
            if hash_index == 1:
                dict1[value] = row
            if hash_index == 2:
                dict2[value] = row
            if hash_index == 3:
                dict3[value] = row
            if hash_index == 4:
                dict4[value] = row
            if hash_index == 5:
                dict5[value] = row
            if hash_index == 6:
                dict6[value] = row
        # transfer list to array, save as .npy file
        alist = np.array(hashlist)
        # print(dict0.keys())
        # save index as npy.file
        if os.path.exists(table + '_' + ''.join(attr) + '_' + index_name + '.npy'):
            raise Exception(
                '[ERROR]: Index  is existed.')
        else:
            np.save(table + '_' + ''.join(attr) + '_' + index_name + '.npy', alist)
        # load index
        # a = np.load('a_index.npy', allow_pickle=True)
        # a = a.tolist()
        print("Index created successfully")

    def exist_index(self, table, attr, num):
        if os.path.exists('index_info.npy'):
            info = np.load('index_info.npy', allow_pickle=True)
            info = info.tolist()
            for item in info:
                # delete the record in index_info.npy
                if item[0] == table and ''.join(item[1]) == attr:
                    # open the table file with index
                    index_name = item[2]
                    if os.path.exists(table + '_' + ''.join(attr) + '_' + index_name + '.npy'):
                        index_info = np.load(table + '_' + ''.join(attr) + '_' + index_name + '.npy', allow_pickle=True)
                        index_info = index_info.tolist()
                        hashtable = HashTable(7)
                        hash_num = hashtable.hash(num)
                        if hash_num == 0:
                            return index_info[0]
                        if hash_num == 1:
                            return index_info[1]
                        if hash_num == 2:
                            return index_info[2]
                        if hash_num == 3:
                            return index_info[3]
                        if hash_num == 4:
                            return index_info[4]
                        if hash_num == 5:
                            return index_info[5]
                        if hash_num == 6:
                            return index_info[6]
            return None

    def drop_index(self, index_name, table):
        # check if index exists
        if os.path.exists('index_info.npy'):
            info = np.load('index_info.npy', allow_pickle=True)
            info = info.tolist()
            # check if the index was existed, index is the third attribute
            i = 0
            for item in info:
                # delete the record in index_info.npy
                if (item[2] == index_name):
                    attr = item[1]
                    info.remove(item)
                    i = i + 1
                    infoarr = np.array(info)
                    np.save('index_info.npy', infoarr)
                    # delete the index.npy file
                    if os.path.exists(table + '_' + ''.join(attr) + '_' + index_name + '.npy'):
                        print(table + '_' + ''.join(attr) + '_' + index_name + '.npy')
                        print(info)
                        os.remove(table + '_' + ''.join(attr) + '_' + index_name + '.npy')
                    else:
                        raise Exception('[ERROR]: The index does not exist')

        else:
            raise Exception('[ERROR]: The index does not exist')

    def index_search(self, attrs, condition):
        '''
        attrs: [attr1, attr2, ...]
        condition:{attr: , value:, operation, }
        '''
        attr=condition['attr']
        value=condition['value']
        operation=condition['operation']
        idex_name=self.index[attr]
        BTree=self.BTree[idex_name]

        min_key=BTree.minKey()
        max_key=BTree.max_key()

        if operation=='=':
            pks=[BTree[value]]
            content=[]
            for pk in pks:
                content.append(self.data[pk])
        
        elif operation=='<':
            pks=list(BTree.values(min=min_key, max=value, excludemin=False, excludemax=True)) #[pk1, pk2,...]
            content=[]
            for pk in pks:
                content.append(self.data[pk])

        elif operation=='>':
            pks=list(BTree.values(min=value, max=max_key, excludemin=True, excludemax=False)) #[pk1, pk2,...]
            content=[]
            for pk in pks:
                content.append(self.data[pk])

        elif operation=='<=':
            pks=list(BTree.values(min=min_key, max=value, excludemin=False, excludemax=False)) #[pk1, pk2,...]
            content=[]
            for pk in pks:
                content.append(self.data[pk])

        elif operation=='>=':
            pks=list(BTree.values(min=value, max=max_key, excludemin=True, excludemax=False)) #[pk1, pk2,...]
            content=[]
            for pk in pks:
                content.append(self.data[pk])

        elif operation=='<>':
            pk1=list(BTree.values(min=min_key, max=value, excludemin=False, excludemax=True)) #[pk1, pk2,...]
            pk2=list(BTree.values(min=value, max=max_key, excludemin=True, excludemax=False)) #[pk1, pk2,...]
            content=[]
            for pk in pk1:
                content.append(self.data[pk])    
            for pk in pk2:
                content.append(self.data[pk])   
        att=self.attrls[len(pks):]

        df=pd.DataFrame(content, columns=att)
        my_df=pd.DataFrame()
        for i in attrs:
            my_df[i]=df[i]
        return my_df

    def insert(self, attrs: list, data: list) -> None:
        """
        Add data into self.data as a hash table.
        TODO:
        -Put data into self.data{} as hash table. Key is prmkvalue, and value is attvalue = [].
        -Use attribute_object.typecheck to check every value, and if the value is invalid, raise error. If not put into attvalu.
        -Check primary key value, if the value already in prmkvalue, raise error.
        -Print essential information
        """
        # TODO: typecheck?
        prmkvalue = []
        attvalue = []
        if attrs==[]:
            # TODO: typecheck
            # Must enter full-attr values by order
            if len(data)!=len(self.attrls):
                raise Exception('[ERROR]: Full-attr values is needed')

            dat = data[::]
            # Get primary-key values
            for _ in range(len(self.primary)):
                prmkvalue.append(dat.pop(0))
            # the remaining data is attr data
            attvalue=dat

            # TODO: typecheck
            for i in data:
                value = data[i]
                attname = self.attrls[i]
                # typecheck()
                # If false, raise error in typecheck()
                # If true, nothing happens and continue
                # If unique, call self uniquecheck()
                if attname in self.uniqueattr.keys():
                    if value in self.uniqueattr[attname].keys():
                        raise Exception('[ERROR]: Unique attribute values are in conflict!  ' + attname + " : " + str(value))
                    self.uniqueattr[attname][value] = prmkvalue
                self.attrs[attname].typecheck(value)
                # If it is not unique, raise [ERROR] in the function
                # Else, continue

            # Hash data
            self.data[tuple(prmkvalue)]=attvalue
        else:
            # Reorder by the oder of self.attrs
            attrs_dict=dict()
            for name in self.attrls:
                attrs_dict[name]=None

            # Get primary-key values
            for name in self.primary:
                if name not in attrs:
                    raise Exception('[ERROR]: Primary key cannot be NULL.')
                prmkvalue.append(data[attrs.index(name)])

            for i in range(len(attrs)):
                value = data[i]
                attname = attrs[i]

                if attname in self.uniqueattr.keys():
                    if value in self.uniqueattr[attname].keys():
                        raise Exception('[ERROR]: Unique attribute values are in conflict!  ' + attname + " : " + str(value))
                    self.uniqueattr[attname][value] = prmkvalue
                #self.attrs[attname].typecheck(value)
                self.attrs[attname].typecheck(value)

                attrs_dict[attname] = value
             # Get primary-key values
            for name in self.primary:
                # Pop primary-key value from the full-attr dict
                attrs_dict.pop(name)
            # The remaining data is attr data
            attvalue=list(attrs_dict.values())

            # Hash data
            if tuple(prmkvalue) not in self.data.keys():
                self.datalist = self.datalist + [prmkvalue + attvalue]
                self.data[tuple(prmkvalue)] = attvalue
            else:
                raise Exception('[ERROR]: Primary key value collision')
        
    def serialize(self):
        pass
    
    def deserialize(self):
        pass
    
    def delete(self, table_name, where):
        if where == []:
            self.data = {}
            self.datalist = []
            for a in self.uniqueattr.keys():
                self.uniqueattr[a] = {}
            #self.BTree
        elif len(where) > 1:
            raise Exception('[ERROR]: Mutiple where conditions is coming soon')
        elif len(where) == 1:
            if where[0]['attr'] not in self.primary:
                raise Exception('[ERROR]: You should delete by one of the primary key!')
            else:
                if where[0]['operation']=='=':
                    value=where[0]['value']
                    try:
                        value=int(value)
                    except:
                        pass
                    # TODO add delete
                    del self.data[tuple([value])]
                    colindex=0
                    keylist=list(self.attrs.keys())
                    while where[0]['attr']!=keylist[colindex]:
                        colindex+=1

                    i=0
                    for item in self.datalist:
                        if item[colindex]==value:
                            break
                        i+=1
                    del self.datalist[i]
                    
                elif where[0]['operation']=='<>':
                    value = where[0]['value']
                    try:
                        value=int(value)
                    except:
                        pass
                    self.data={self.data[value]}
                    self.df=self.df[self.df[where[0]['attr']]==value]

    # res = table.search('*', '=', False, ['id', 5], False)
    # tag: is str
    # 'condition': [  where[i]['attr'], where[i]['value']  ]
    def search(self, attr, sym, tag, condition, groupbyFlag):
        # attr: [] or *
        # situation: number means different conditions
        # groupbyFlag: true/false have group by
        # condition: [], base on situation
        # df = pd.DataFrame(self.datalist, columns = self.attrls)
        if self.flag == 0:
            self.df = pd.DataFrame(self.datalist, columns = self.attrls)
        operationList = {
            '=': 1,
            '>': 2,
            '>=': 3,
            '<': 4,
            '<=': 5,
            'LIKE': 6,
            'NOT LIKE': 7,
            '<>': 8
        }
        if len(sym) == 0:
            situation = 0
        else:
            situation = operationList[sym]
        if groupbyFlag:
            temp = self.group_by(condition[2], condition[3], attr, df)
        else:
            temp = self.df

        if situation == 0:  # no where
            if attr == ['*']:
                return temp
            else:
                return temp.loc[:, attr]

        if situation == 1:
            ###################
            # get index
            ##################
            # TODO use index
            gettable=self.exist_index(self.name, condition[0], condition[1])
            if gettable is not None:
                # has index
                templist=[]
                for item in gettable.values():
                    templist.append(item)
                data=pd.DataFrame(templist,columns=self.df.columns)
                temp=data
            else:
                pass
            if tag:
                if attr == ['*']:
                    return temp.loc[temp[condition[0]] == temp[condition[1]]]
                #
                return temp.loc[temp[condition[0]] == temp[condition[1]], attr]
            if attr == ['*']:
                return temp.loc[temp[condition[0]] == condition[1]]
            #
            return temp.loc[temp[condition[0]] == condition[1], attr]
        if situation == 2:
            if tag:
                if attr == ['*']:
                    return temp.loc[temp[condition[0]] > temp[condition[1]]]
                return temp.loc[temp[condition[0]] > temp[condition[1]], attr]
            if attr == ['*']:
                return temp.loc[temp[condition[0]] > condition[1]]
            return temp.loc[temp[condition[0]] > condition[1], attr]
        if situation == 3:
            if tag:
                if attr == ['*']:
                    return temp.loc[temp[condition[0]] >= temp[condition[1]]]
                return temp.loc[temp[condition[0]] >= temp[condition[1]], attr]
            if attr == ['*']:
                return temp.loc[temp[condition[0]] >= condition[1]]
            return temp.loc[temp[condition[0]] >= condition[1], attr]
        if situation == 4:
            if tag:
                if attr == ['*']:
                    return temp.loc[temp[condition[0]] < temp[condition[1]]]
                return temp.loc[temp[condition[0]] < temp[condition[1]], attr]
            if attr == ['*']:
                return temp.loc[temp[condition[0]] < condition[1]]
            return temp.loc[temp[condition[0]] < condition[1], attr]
        if situation == 5:
            if tag:
                if attr == ['*']:
                    return temp.loc[temp[condition[0]] <= temp[condition[1]]]
                return temp.loc[temp[condition[0]] <= temp[condition[1]], attr]
            if attr == ['*']:
                return temp.loc[temp[condition[0]] <= condition[1]]
            return temp.loc[temp[condition[0]] <= condition[1], attr]
        if situation == 8:
            if tag:
                if attr == ['*']:
                    return temp.loc[temp[condition[0]] != temp[condition[1]]]
                return temp.loc[temp[condition[0]] != temp[condition[1]], attr]
            if attr == ['*']:
                return temp.loc[temp[condition[0]] != condition[1]]
            return temp.loc[temp[condition[0]] != condition[1], attr]

        if situation == 6:
            if tag:
                if attr == ['*']:
                    return temp.loc[temp[condition[0]].str.contains(temp[condition[1]])]
                return temp.loc[temp[condition[0]].str.contains(temp[condition[1]]), attr]
            if attr == ['*']:
                return temp.loc[temp[condition[0]].str.contains(condition[1])]
            return temp.loc[temp[condition[0]].str.contains(condition[1]), attr]
        if situation == 7:
            if tag:
                if attr == ['*']:
                    return temp.loc[~temp[condition[0]].str.contains(temp[condition[1]])]
                return temp.loc[~temp[condition[0]].str.contains(temp[condition[1]]), attr]
            if attr == ['*']:
                return temp.loc[~temp[condition[0]].str.contains(condition[1]), attr]
            return temp.loc[~temp[condition[0]].str.contains(condition[1]), attr]


    def group_by(self, agg, attr_gr, attr, df):
        """
        :param situation: calculation of group by
        :param attr_gr: the attrs for group
        :param attr: attrs for calculation
        :param df: dataframe for group by
        :return: a dataframe
        """
        agg_funcs = {
            'MAX': 0,
            'MIN': 1,
            'AVG': 2,
            'SUM': 3,
            'COUNT': 4
        }
        situation = agg_funcs[agg]

        if attr == '*':
            raise Exception('[ERROR]: Invalid search.')
        gb = df.groupby(attr_gr)
        if situation == 0:
            return gb[attr].max()
        if situation == 1:
            return gb[attr].min()
        if situation == 2:
            return gb[attr].mean()
        if situation == 3:
            return gb[attr].sum()
        if situation == 4:
            return gb[attr].value_counts()

    def table_join(self, table, attr):
        df1 = pd.DataFrame(self.data)
        df2 = pd.DataFrame(table.data)
        return pd.merge(df1, df2, on=attr)


# if __name__ == '__main__':
#     data = []
#     for i in range(100):
#         if i > 50:
#             data.append([i,2])
#         else:
#             data.append([i,1])
#     attr1 = {'name': 'id', 'type': 'INT', 'notnull': False, 'unique': False}
#     attr2 = {'name': 'num', 'type': 'INT', 'notnull': False, 'unique': False}
#     info = {'name': 'test', 'attrs': [attr1, attr2], 'primary': '', 'foreign': []}
#     table = Table(['id', 'num'], info)
#     table.df = pd.DataFrame(data, columns=['id', 'num'])
#     print(table.df)
#     res = table.search('*', '=', False, ['id', 5], False)
#     # print(res)
