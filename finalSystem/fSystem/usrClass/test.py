def KMeans(fileName, n):
    from sklearn.cluster import KMeans
    from sklearn.externals import joblib

    wordList = ['堆放','杂物','垃圾','咨询']
    feature = [[0.2,0.3],[0.6,0.1],[0.5,0.3],[0.7,0.5],]

    # 调用kmeans类
    clf = KMeans(n_clusters=n)
    s = clf.fit(feature)
    print (s)

    # 9个中心
    print (clf.cluster_centers_)

    # 每个样本所属的簇
    print (clf.labels_)

    # 用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
    print (clf.inertia_)

    # 进行预测
    print (clf.predict(feature))

    # 保存模型
    #joblib.dump(clf,  'km.pkl')

    # 载入保存的模型
    # clf = joblib.load('c:/km.pkl')

    '''
    #用来评估簇的个数是否合适，距离越小说明簇分的越好，选取临界点的簇个数
    for i in range(5,30,1):
        clf = KMeans(n_clusters=i)
        s = clf.fit(feature)
        print i , clf.inertia_
    '''
KMeans('',2)