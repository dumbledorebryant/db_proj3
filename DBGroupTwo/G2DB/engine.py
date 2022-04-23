import os
import sys
import pickle
import shutil

import csv
import json
import pandas

from G2DB.parser import startParse
from G2DB.core.database import Database
from G2DB.core.table import Table


# All attributes, table names, and database names will be stored in lower case

# database engine
class Engine:

    # action functions
    # CREATE DATABASE test;
    def createDatabase(self, name):
        db = Database(name)
        print('[SUCCESS]: Create Database %s successfully. You also need to save it.' % name)
        return db

    # save database test;
    def saveDatabase(self, db):
        db.save()

    # DROP DATABASE test;
    def dropDatabase(self, db):
        db.drop_database()

    # use database test;
    def useDatabase(self, dbname):
        sysPath = sys.argv[0]
        sysPath = sysPath[:-11] + 'database/'

        if not os.path.exists(sysPath + dbname):
            raise Exception('[ERROR]: No Database called %s.' % dbname)

        elif os.path.exists(sysPath + dbname):
            file = open(sysPath + dbname, "rb")
            db = pickle.load(file)
            file.close()
            print('[SUCCESS]: Open Database %s successfully!' % dbname)

        else:
            raise Exception('[ERROR]: Invalid command.')
        return db

    # show databases;
    def show_database(self):
        t = sys.argv[0]
        t = t[:-11] + 'database/'
        dirs = os.listdir(t)
        for dir in dirs:
            if '.' not in dir:
                print(dir)

    # show tables;
    def show_table(self, db):
        db.display()

    """
    CREATE TABLE table_name (column_name1 data_type not_null, column_name2 data_type null) primary_key (column_name1);
    CREATE TABLE table_name (column_name1 data_type not_null, column_name2 data_type null) primary_key (column_name1, column_name2) foreign_key (column_name_f, column_namef1) references database_name.table_name (column_name);
    CREATE TABLE table_name (column_name1 data_type not_null unique, column_name2 data_type null) primary_key (column_name1, column_name2) foreign_key (column_name_f, column_namef1) references database_name.table_name (column_name);
    """

    def createTable(self, db, attrs, info):
        db.add_table(attrs, info)
        print('[SUCCESS]: Create Table %s successfully!' % info['name'])
        return db

    # DROP TABLE a;
    def dropTable(self, db, table_name):
        db.drop_table(table_name)

    # insert into perSON (id, position, name, address) values (2, 'eater', 'Yijing', 'homeless')
    def insertTable(self, db, table_name, attrs, data):
        db.tables[table_name].insert(attrs, data)
        print(db.tables[table_name].datalist)
        return db

    def selectQuery(self, db, attrs, tables, where):
        # Return restable
        """
        ats = list(attrs.keys())
        table = db.tables[tables[0]]
        if where:
            cond = {'tag': False, 'sym': where[0]['operation'], 'condition': [where[0]['attr'],  where[0]['value']]}
        else:
            cond = {}
        restable = self.subselect(table, ats, cond)
        return restable
        """
        table_col_dic = {}
        attrs = list(attrs.keys())

        if attrs != ["*"]:
            # table_col_dic----> {'tableName': ['colName','colName'...]}
            for table_name in tables:
                colNameList = []
                for attr in attrs:
                    if attr in db.tables[table_name].attrls:
                        colNameList.append(attr)

                table_col_dic[table_name] = colNameList

        else:
            # table_col_dic----> {'tableName': ['*']}
            for table_name in tables:
                table_col_dic[table_name] = ['*']

        tc = []
        used_attrs = []
        join_con = []
        # where-->{'attr': 'a', 'value': 5.0, 'operation': '>=', 'tag': 0}
        # print('==', where)
        for item in where:
            # print(item['tag'])
            if item['tag'] == 1:

                join_con.append(item)
                where.remove(item)

        tbl = tables[::]

        # print ("join_con")
        # print(join_con)
        if join_con:
            condition = join_con.pop(0)
            # print ("condition")
            # print(condition)

            print(db.tables[table_name])
            print(db.tables[table_name].attrls)
            exit()
            # Get first three elems 
            for table_name in tables:
                if condition['attr'] in db.tables[table_name].attrls:
                    tc.append(table_name)
                    tbl.remove(table_name)
                    used_attrs = used_attrs + db.tables[table_name].attrls
                    if (condition['attr'] not in table_col_dic[table_name]) & (table_col_dic[table_name] != ['*']):
                        table_col_dic[table_name].append(condition['attr'])

            for table_name in tables:
                if condition['value'] in db.tables[table_name].attrls:
                    tc.append(table_name)
                    tbl.remove(table_name)
                    used_attrs = used_attrs + db.tables[table_name].attrls
                    if (condition['attr'] not in table_col_dic[table_name]) & (table_col_dic[table_name] != ['*']):
                        table_col_dic[table_name].append(condition['value'])
                    tc.append(condition)

        # Append one table and one condition by order
        while join_con:
            for condition in join_con:
                if condition['attr'] in used_attrs:
                    for table_name in tables:
                        if condition['value'] in db.tables[table_name]:
                            tc.append(table_name)
                            tbl.remove(table_name)
                            used_attrs = used_attrs + db.tables[table_name].attrls
                            tc.append(condition)
                            join_con.remove(condition)
                            if (condition['attr'] not in table_col_dic[table_name]) & (
                                    table_col_dic[table_name] != ['*']):
                                table_col_dic[table_name].append(condition['value'])
                    continue
                elif condition['value'] in used_attrs:
                    for table_name in tables:
                        if condition['attr'] in db.tables[table_name]:
                            tc.append(table_name)
                            tbl.remove(table_name)
                            used_attrs = used_attrs + db.tables[table_name].attrls
                            tc.append(condition)
                            join_con.remove(condition)
                            if (condition['attr'] not in table_col_dic[table_name]) & (
                                    table_col_dic[table_name] != ['*']):
                                table_col_dic[table_name].append(condition['attr'])
                    continue

                else:
                    raise Exception('[ERROR]: Wrong command.')

        while tbl:
            if len(tc) > 3:
                tc.append(tbl[0])
                tc.append({})
                tbl.pop(0)
            elif (len(tc) == 0) & (len(tbl) > 1):
                tc.append(tbl[0])
                tc.append(tbl[1])
                tc.append({})
                tbl.pop(0)
                tbl.pop(0)
            else:
                tbl.pop(0)

        vc = where

        print({
            'ta': table_col_dic,
            'tc': tc,
            'vc': where,

        })

        if tc:

            to = {}
            for tname in table_col_dic.keys():
                to[tname] = self.subselect(db.tables[tname], table_col_dic[tname], [])

            if tc[2]:
                jointable = self.join(to[tc[0]], to[tc[1]], [tc[2]['attr'], tc[2]['value']])
            else:
                jointable = self.join(to[tc[0]], to[tc[1]], [])
            tc.pop(0)
            tc.pop(0)
            tc.pop(0)
            while tc:
                jointable = self.join(jointable, to[tc[0]], [tc[1]['attr'], tc[1]['value']])
                tc.pop(0)
                tc.pop(0)

            info = {'name': 'test', 'attrs': [], 'primary': '', 'foreign': []}
            table = Table(jointable.columns, info)
            table.df = jointable
            table.flag = 1

        elif len(tables) == 1:
            table = db.tables[tables[0]]

        if vc:
            cond = {'tag': vc[0]['tag'], 'sym': vc[0]['operation'], 'condition': [vc[0]['attr'], vc[0]['value']]}
        else:
            cond = {}

        restable = self.subselect(table, attrs, cond)
        return restable

    def subselect(self, table, attrs, where):
        sym = ''
        tag = False
        gb = False
        condition = []
        if where:
            sym = where['sym']
            tag = where['tag']
            condition = where['condition']
            df = table.search(attrs, sym, tag, condition, gb)
        else:
            df = table.search(attrs, sym, tag, condition, gb)
        return df

    def join(self, table1, table2, attrs):
        df = Database('jointempdb').join_table(table1, table2, attrs)
        return df

    def addor(self, table1, table2, ao):
        if ao == "0":
            df = Database('jointempdb').df_(table1, table2, attr)
        return df

    def delete(self, db, name, where):
        db.tables[name].delete(name, where)
        return db

    def update(self, db, name, where, set):
        db = self.delete(db, name, where)
        db = self.insertTable(db, name, set['attrs'], set['data'])
        return db

    def createIndex(self, db, table, iname, attr):
        db = db.tables[table].add_index(attr, iname)
        return db

    def dropIndex(self, db, table, iname):
        db = db.tables[table].drop_index(iname)
        return db

    # lauch function: receieve a command and send to execution function.
    def start(self):
        db = None
        # continue running until recieve the exit command.
        while True:
            inputstr = 'GroupTwo>'
            if db:
                inputstr = db.name + '> '
            else:
                inputstr = 'GroupTwo> '
            commandline = input(inputstr)
            ########################
            # Load Test Data
            ########################
            if commandline == 'load demo data':
                demoQuery = 'create database demo;\n'
                testNum = 0
                while testNum < 5:
                    # define data range
                    dataRange = 1000
                    if testNum == 2:
                        dataRange = 10000
                    elif testNum == 4:
                        dataRange = 100000

                    # Rel-i-i-dataRange
                    tableName = 'test' + str(testNum)
                    col1name = 'a'
                    col2name = 'b'
                    demoQuery += 'create table ' + tableName + ' (' + col1name + ' int not_null unique, ' + col2name + ' int unique) primary key (' + col1name + ');\n'
                    for i in range(dataRange + 1):
                        demoQuery += 'insert into ' + tableName + ' (' + col1name + ', ' + col2name + ') values (' + str(
                            i) + ', ' + str(i) + ');\n'
                    testNum += 1

                    # Rel-i-1-dataRange
                    tableName = 'test' + str(testNum)
                    demoQuery += 'create table ' + tableName + ' (' + col1name + ' int not_null unique, ' + col2name + ' int) primary key (' + col1name + ');\n'
                    for i in range(dataRange + 1):
                        demoQuery += 'insert into ' + tableName + ' (' + col1name + ', ' + col2name + ') values (' + str(
                            i) + ', 1);\n'
                    testNum += 1
                demoQuery += 'save database demo;\n'
                # create query over
                # run demo query
                commandlines = demoQuery.split(';\n')
                for commandline in commandlines:
                    # ignore enter and ;
                    if commandline == '':
                        continue
                    commandline = commandline.replace(';', '')
                    # run query
                    try:
                        result, db = self.execute(commandline, db)
                        if result == 'exit':
                            print('[SUCCESS]: Exit successfully! See you next time!')
                            sys.exit(0)
                            return
                    # print error
                    except Exception as err:
                        print(err)
                continue

            ########################
            # Run input query
            ########################
            # ignore enter and ;
            if commandline == '':
                continue
            commandline = commandline.replace(';', '')
            # run query
            try:
                result, db = self.execute(commandline, db)
                if result == 'exit':
                    print('[SUCCESS]: Exit successfully! See you next time!')
                    sys.exit(0)
                    return
            except Exception as err:
                print(err)

    # execution function: send commandline to parser and get an action as return and execute the mached action function.
    def execute(self, commandline, database):
        # parse the query
        db = database
        action = startParse(commandline)

        if action['query_keyword'] == 'exit':
            return 'exit', db
        ########################
        # CREATE
        ########################
        if action['query_keyword'] == 'create':
            if action['type'] == 'database':
                db = self.createDatabase(action['name'])
            elif action['type'] == 'table':
                if db:
                    if action['name'] in db.tables.keys(): raise Exception(
                        '[ERROR]: Table %s is exsited.' % action['name'])
                    db = self.createTable(db, action['attrls'], action['info'])
                else:
                    raise Exception('[ERROR]: No database name in command/ Cannot find database name.')
            elif action['type'] == 'index':
                if db:
                    db = self.createIndex(db, action['table'], action['index_name'], action['attrs'])
                else:
                    raise Exception('[ERROR]: No database name in command/ Cannot find database name.')
            self.saveDatabase(db)
            return 'continue', db
        ########################
        # DROP
        ########################
        if action['query_keyword'] == 'drop':
            if action['type'] == 'database':
                if db:
                    if action['name'] == db.name:
                        self.dropDatabase(db)
                        db = None
                    else:
                        temp = db
                        db = Database(action['name'])
                        self.dropDatabase(db)
                        db = temp
                else:
                    db = Database(action['name'])
                    self.dropDatabase(db)
                    db = None
            elif action['type'] == 'table':
                self.dropTable(db, action['table_name'])
            elif action['type'] == 'index':
                if db:
                    db = self.dropIndex(db, action['table'], action['index_name'])
                else:
                    raise Exception('[ERROR]: No database name in command/ Cannot find database name.')
            self.saveDatabase(db)
            return 'continue', db
        ########################
        # INSERT
        ########################
        if action['query_keyword'] == 'insert':
            db = self.insertTable(db, action['table_name'], action['attrs'], action['data'])
            self.saveDatabase(db)
            return 'continue', db
        ########################
        # SELECT
        ########################
        if action['query_keyword'] == 'select':

            if db:
                restable = self.selectQuery(db, action['attrs'], action['tables'], action['where'])
                print(restable)
                return 'continue', db
            else:
                raise Exception('[ERROR]: No database name in command/ Cannot find database name.')

        ########################
        # DELETE
        ########################
        if action['query_keyword'] == 'delete':
            db = self.delete(db, action['table'], action['where'])
            self.saveDatabase(db)
            return 'continue', db

        ########################
        # UPDATE
        ########################
        if action['query_keyword'] == 'update':
            db = self.update(db, action['table'], action['where'], action['set'])
            self.saveDatabase(db)
            return 'continue', db

        # ########################
        # # SAVE
        # ########################
        # if action['query_keyword'] == 'save':
        #     if db:
        #         if action['name'] == db.name:
        #             self.saveDatabase(db)
        #         else:
        #             raise Exception('[ERROR]: No database called %s.' % db.name)
        #     else:
        #         raise Exception('[ERROR]: No database called %s.' % db.name)
        #     return 'continue', db

        ########################
        # USE
        ########################
        if action['query_keyword'] == 'use':
            db = self.useDatabase(action['name'])
            return 'continue', db

        ########################
        # SHOW
        ########################
        if action['query_keyword'] == 'show':
            if action['type'] == 'database':
                self.show_database()
            else:
                self.show_table(db)
            return 'continue', db
