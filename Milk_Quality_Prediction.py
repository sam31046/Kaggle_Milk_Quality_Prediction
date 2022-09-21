#!/usr/bin/env
# -*- coding: utf-8 -*-
__author__ = "Jhong, Dong-You"


# To import chinese font on Linus, MacOS, Windows
# For GUI, plot purpose
# import sys
class font_import(object):

    def __init__(self):
        import sys
        if sys.platform.startswith("linux"):
            # could be "linux", "linux2", "linux3", ...
            self.font_linuxOS()
        elif sys.platform == "darwin":
            # MAC OS X
            self.font_macOS()
        elif sys.platform == "win32":
            # Windows (either 32-bit or 64-bit)
            self.font_winOS()

    def font_linuxOS(self):
        print("linux need font")  # linux
        print("Initiated font")

    def font_macOS(self):
        try:
            import seaborn as sns
            sns.set(font="Arial Unicode MS")  # "DFKai-SB"
            print("Initiated Seaborn font")
        except:
            print("Initiated Seaborn font failed")
        try:
            import matplotlib.pyplot as plt
            from matplotlib.font_manager import FontProperties
            plt.rcParams['font.sans-serif'] = 'Arial Unicode MS'
            plt.rcParams['axes.unicode_minus'] = False
            print("Initiated matplotlib font")
        except:
            print("Initiated matplotlib font failed")

    def font_winOS(self):
        # Windows (either 32-bit or 64-bit)
        try:
            import seaborn as sns
            sns.set(font="sans-serif")  # "DFKai-SB"
            print("Initiated Seaborn font ")
        except:
            print("Initiated Seaborn font failed")
        try:
            import matplotlib.pyplot as plt
            plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
            plt.rcParams['axes.unicode_minus'] = False
            print("Initiated matplotlib font")
        except:
            print("Initiated matplotlib font failed")


def Dataframe_save_XLSX(dataframe, filename='.xlsx'):
    # Save the excelsheet
    dataframe.to_excel(filename)
    print('Save XLSX successfully!')


def Dataframe_read_Excel(filename='.xlsx'):
    # Read the excelsheet
    dataframe = pd.read_excel(filename, sheet_name=0)
    print('Read XLSX successfully!')
    return dataframe


def Dataframe_Info(dataframe):
    # Data info
    print("*** headers: \n", list(dataframe))  # The most efficient way to get df header list
    print("*** type: ", type(dataframe))
    print("*** dtypes: ", dataframe.dtypes)
    print("*** shape: ", dataframe.shape)
    print("*** columns: ", dataframe.columns)
    print("*** index: ", dataframe.index)
    print("*** info: ", dataframe.info)
    print("*** describe: ", dataframe.describe)


# Exclude longitude and latitude outliers.
def Dataframe_outlier_exclude(dataframe, column, outlier_value=999999):
    df_raw = dataframe
    df_filtered = df_raw[
        df_raw[column] != outlier_value
        ]
    return df_filtered


# Show unique values of every column to check whether nan or outliers exist.
def Dataframe_show_unique_cell(dataframe):
    print("*** Unique values of columns: ")
    for col_name in list(dataframe):
        print(col_name + ':', dataframe[col_name].unique())


# Replace string, including headers.
def Dataframe_replace_list_by_list(dataframe, target_word_list, replace_word_list):
    # Replace special characters:
    # df.columns = df.columns.str.replace('[ ,#,@,!,&,&,%,?,/,\,+,~,<,>]', '', regex=True)
    for target_word, replace_word in zip(target_word_list, replace_word_list):
        # print(type(target_word),type(replace_word))
        dataframe.columns = dataframe.columns.str.replace(target_word, replace_word)
    return dataframe


# Normalization: MinMaxScaler
def Dataframe_Norm_MinMaxScaler(X_data):
    from sklearn import preprocessing
    mmscaler = preprocessing.MinMaxScaler()
    X_data_minmax = mmscaler.fit_transform(X_data)  # Range: 0 to 1　
    print(X_data)
    print(X_data_minmax)
    return X_data_minmax


# Scatter plots
def Dataframe_Scatter_Plot(dataframe):
    # Scatter plots
    sns.scatterplot(x="pH", y="Temperature", hue="Grade", data=dataframe)
    plt.show()
    sns.scatterplot(x="pH", y="Turbidity", hue="Grade", data=dataframe)
    plt.show()

    sns.scatterplot(x="Temperature", y="Turbidity", hue="Grade", data=dataframe)
    plt.show()
    sns.scatterplot(x="Temperature", y="Fat", hue="Grade", data=dataframe)
    plt.show()
    sns.scatterplot(x="Temperature", y="Odor", hue="Grade", data=dataframe)
    plt.show()

    sns.scatterplot(x="Taste", y="Fat", hue="Grade", data=dataframe)
    plt.show()

    sns.scatterplot(x="Odor", y="Turbidity", hue="Grade", data=dataframe)
    plt.show()
    sns.scatterplot(x="Odor", y="Fat", hue="Grade", data=dataframe)
    plt.show()

    sns.scatterplot(x="Fat", y="Turbidity", hue="Grade", data=dataframe)
    plt.show()


# Correlation plots, methods:{‘pearson’, ‘kendall’, ‘spearman’}
def Dataframe_Correlation_Plot(dataframe):
    # Methods:{‘pearson’, ‘kendall’, ‘spearman’}
    corr_pearson = dataframe.corr(method='pearson')
    corr_kendall = dataframe.corr(method='kendall')
    corr_spearman = dataframe.corr(method='spearman')

    # Corr: Pearson
    sns.heatmap(corr_pearson, cmap="YlGnBu")
    plt.title('Pearson Correlation', fontsize=10)
    plt.xticks(fontsize=7, rotation=30)
    plt.yticks(fontsize=7, rotation=30)
    # plt.xlabel(fontsize=10, rotation=45)
    # plt.ylabel(fontsize=10, rotation=45)
    plt.show()

    # Corr: Kendall
    sns.heatmap(corr_kendall, cmap="YlGnBu")
    plt.title('Kendall Correlation', fontsize=10)
    plt.xticks(fontsize=7, rotation=30)
    plt.yticks(fontsize=7, rotation=30)
    # plt.xlabel(fontsize=10, rotation=45)
    # plt.ylabel(fontsize=10, rotation=45)
    plt.show()

    # Corr: Spearman
    sns.heatmap(corr_spearman, cmap="YlGnBu")
    plt.title('Spearman Correlation', fontsize=10)
    plt.xticks(fontsize=7, rotation=30)
    plt.yticks(fontsize=7, rotation=30)
    # plt.xlabel(fontsize=10, rotation=45)
    # plt.ylabel(fontsize=10, rotation=45)
    plt.show()


# Sns pair plots
def Dataframe_Pair_Plot(dataframe):
    sns.set_style('whitegrid')
    sns.pairplot(dataframe, hue='Grade', height=2)
    plt.show()


# Histogram plots
def Dataframe_Hist_Plot(dataframe, title='Unique Value Count'):
    import matplotlib.pyplot as plt
    dataframe.hist()
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def ML_Split_Data(dataframe, label_col='', test_size=0.1):
    from sklearn.model_selection import train_test_split
    # feature data are columns without label(class) column
    # For exclude many columns, df.loc[:, ~df.columns.isin(['rebounds', 'assists'])]
    feature_data = dataframe.iloc[:, dataframe.columns != label_col]
    labels = dataframe[label_col].to_numpy()  # df.to_numpy() == df.values
    # Test samples: test_size * 100% of the data
    train_x, test_x, train_y, test_y = train_test_split(feature_data, labels, test_size=test_size)
    return train_x, test_x, train_y, test_y


def ML_Random_Forest(dataframe, label_col='', plot=True):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import export_graphviz
    from sklearn import tree
    # Split data
    train_x, test_x, train_y, test_y = ML_Split_Data(dataframe, label_col)
    # Random Forest
    features = list(dataframe)[0:7]
    category = dataframe[label_col].unique()
    category = list(map(str, category))
    rf = RandomForestClassifier(n_estimators=100,
                                random_state=2,
                                max_depth=7)
    rf.fit(train_x, train_y)
    prediction = rf.predict(test_x)
    rfScore = rf.score(test_x, test_y)
    print("Random Forest predict answer：", prediction, " Accuracy：", rfScore)

    if plot is True:
        export_graphviz(rf.estimators_[2], out_file='Random_Forest.dot',
                        feature_names=features,
                        class_names=category,
                        rounded=True, proportion=False,
                        precision=2, filled=True)

        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(100, 50))

        for index in range(0, 5):
            tree.plot_tree(rf.estimators_[index],
                           feature_names=features,
                           class_names=category,
                           filled=True,
                           ax=axes[index],
                           # Display maximum
                           # max_depth=4,
                           fontsize=8)

            axes[index].set_title('Estimator: ' + str(index), fontsize=20)
        fig.savefig('Random_Forest1.png', dpi=200, format='png')
        print('Random Forest Figure saved.')
        plt.show()
    else:
        pass
    return rfScore


def ML_Decision_Tree(dataframe, label_col='', plot=True):
    from sklearn import tree
    # Split data
    train_x, test_x, train_y, test_y = ML_Split_Data(dataframe, label_col)
    # Decision Tree
    features = list(dataframe)[0:7]
    category = dataframe[label_col].unique()
    category = list(map(str, category))
    clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=3)
    clf = clf.fit(train_x, train_y)
    prediction = clf.predict(test_x)
    clfScore = clf.score(test_x, test_y)
    print("Decision Tree predict answer：", prediction, " Accuracy：", clfScore)
    # Plot
    if plot is True:
        tree.export_graphviz(clf, out_file='Decision_Tree.dot', feature_names=features)
        fig = plt.figure(figsize=(7, 7))
        tree.plot_tree(clf,
                       feature_names=features,
                       class_names=category,
                       filled=True)
        fig.savefig("Decision_Tree1.png", dpi=200, format='png')
        print('Decision Tree Figure saved.')
        plt.show()
    else:
        pass
    return clfScore


# Need to input label column name, because KMeans has to exclude it.
def ML_KMeans(dataframe, label_col):
    from sklearn.cluster import KMeans
    from sklearn import metrics
    # Split data
    train_x, test_x, train_y, test_y = ML_Split_Data(dataframe, label_col)
    # KMeans 演算法
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(train_x)
    y_predict = kmeans.predict(test_x)
    kmeans_score = metrics.accuracy_score(test_y, kmeans.predict(test_x))
    kmeans_homogeneity_score = metrics.homogeneity_score(test_y, kmeans.predict(test_x))
    print("KMeans predict answer：", y_predict, " Accuracy：", kmeans_score)
    print("KMeans predict answer：", y_predict, " Fixed Accuracy：", kmeans_homogeneity_score)
    return kmeans_homogeneity_score


def ML_KNN(dataframe, label_col, neighbors=5, p=1):
    from sklearn.neighbors import KNeighborsClassifier  # pip3 install -U scikit-learn
    # Split data
    train_x, test_x, train_y, test_y = ML_Split_Data(dataframe, label_col)
    # KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=neighbors, p=p)  # 3 5 9
    knn.fit(train_x, train_y)
    knnPredict = knn.predict(test_x)
    knnScore = knn.score(test_x, test_y)
    print("KNN predict answer：", knnPredict, " Accuracy：", knnScore)
    print("Actual Answer：", test_y)
    return knnScore


"""
# Convert to numpy array without index and header
X = dataframe.to_numpy(dtype='float32')
X = ML_PCA(X, 2)
"""


def ML_PCA(X, to_dimension=2):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=to_dimension)
    X_dimension_reduced = pca.fit_transform(X)
    return X_dimension_reduced


"""
X: 2 dimensions only, two features.
# Convert to numpy array without index and header
X = dataframe.to_numpy(dtype='float32')
y = dataframe[label_col].to_numpy(dtype='int')
ML_Decision_Region_mlxtend(X, y, rf)
"""


def ML_Decision_Region_mlxtend(X, y, classifier):
    # y must be an integer array
    from mlxtend.plotting import plot_decision_regions
    plot_decision_regions(X=X, y=y, clf=classifier)


# To One-Hot encode label column
def AI_Encode_One_Hot(train_y, test_y, category):
    import tensorflow as tf
    train_y_2 = tf.keras.utils.to_categorical(train_y, num_classes=category)
    test_y_2 = tf.keras.utils.to_categorical(test_y, num_classes=category)
    return train_y_2, test_y_2


# Tensorflow Keras, Multilayer Perceptron
def AI_MLP_Keras(dataframe, label_column='', hidden_layers=1):
    import tensorflow as tf
    import numpy as np
    df_data = dataframe.iloc[:, dataframe.columns != label_column]
    df_label = dataframe[label_column]
    # Category: 10 classes in label column
    category = df_label.count()
    # Split data
    train_x, test_x, train_y, test_y = ML_Split_Data(dataframe, label_column)
    # To One-Hot encode label column
    train_y_2, test_y_2 = AI_Encode_One_Hot(train_y, test_y, category)
    # Dimension: How many columns each row
    dim = len(list(df_data))

    # Build up model
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(units=50,
                                    activation=tf.nn.relu,
                                    input_dim=dim))
    # Hidden layers
    for hidden_layer in range(hidden_layers):
        model.add(tf.keras.layers.Dense(units=100,
                                        activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=category,
                                    activation=tf.nn.softmax))
    model.compile(optimizer='adam',
                  # This loss is specific for OneHot encoding purpose
                  loss=tf.keras.losses.categorical_crossentropy,
                  metrics=['accuracy'])
    model.fit(train_x, train_y_2,   # Before One-Hot encoding, use train_y instead.
              epochs=2000,
              batch_size=500)

    # Test
    score = model.evaluate(test_x, test_y_2)   # Before One-Hot encoding, use test_y_2 instead.
    print("score:", score)
    predict = model.predict(test_x)
    # print("predict:",predict)
    # print("Ans:",np.argmax(predict[0]),np.argmax(predict[1]),np.argmax(predict[2]),np.argmax(predict[3]))
    print("y_answer:", np.argmax(predict, axis=-1))

    print("y_test", test_y[:])
    return score

# Model file: filename_json.jason
# Weights file: filename_h5.h5
def AI_Save_Model_and_Weights(model, filename_json, filename_h5):
    # Save model .jason
    with open(filename_json + ".json", "w") as json_file:
        json_file.write(model.to_json())
    # Save Weights
    model.save_weights(filename_h5 + ".h5")


if __name__ == '__main__':
    import pandas as pd
    import matplotlib as mpl

    mpl.use("TKAgg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    font_import()

    df = pd.read_csv(filepath_or_buffer='milknew.csv')

    Dataframe_Info(df)

    df.rename(columns={'Temprature': 'Temperature', 'Fat ': 'Fat'}, inplace=True)
    # Label encoding
    df['Grade'] = df['Grade'].map({'high': 3, 'medium': 2, 'low': 1})
    Dataframe_show_unique_cell(df)

    # Normalization, using MinMaxScaler
    import pandas as pd
    from sklearn import preprocessing

    df_raw_data = df.iloc[:, :7]        # Without label column
    print("*** Before Normalization: \n", df_raw_data)
    data_numpy = df_raw_data.values     # returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    data_scaled_numpy = min_max_scaler.fit_transform(data_numpy)
    df_norm_data = pd.DataFrame(data_scaled_numpy)
    print("*** After Normalization: \n", df_norm_data)
    df2 = df_norm_data
    df2 = pd.concat([df2, df['Grade']], axis=1)  # Create a new dataframe with norm values

    # Machine Learning
    # Train and test
    # Train Scores before Normalization
    run_times = 5
    scoreRF = []
    scoreDT = []
    scoreKM = []
    scoreKNN = []
    for run_ime in range(run_times):
        scoreRF.append(100 * ML_Random_Forest(df, 'Grade', plot=False))
        scoreDT.append(100 * ML_Decision_Tree(df, 'Grade', plot=False))
        scoreKM.append(100 * ML_KMeans(df, 'Grade'))
        scoreKNN.append(100 * ML_KNN(df, 'Grade', neighbors=5, p=1))
    print('scoreRF: \n', scoreRF)
    print('scoreDT: \n', scoreDT)
    print('scoreKM: \n', scoreKM)
    print('scoreKNN: \n', scoreKNN)
    score_list = [scoreRF, scoreDT, scoreKM, scoreKNN]
    score_labels = ['Random Forest', 'Decision Tree', 'KMeans', 'KNeighborsClassifier']
    for scores, label in zip(score_list, score_labels):
        plt.plot(scores, label=label)
        plt.xlabel('Run Time(count)')
        plt.ylabel('Score(%)')
    plt.legend()
    plt.title('Train Scores before Normalization')
    plt.show()

    # Train Scores after Normalization
    run_times = 5
    scoreRF = []
    scoreDT = []
    scoreKM = []
    scoreKNN = []
    for run_ime in range(run_times):
        scoreRF.append(100 * ML_Random_Forest(df2, 'Grade', plot=False))
        scoreDT.append(100 * ML_Decision_Tree(df2, 'Grade', plot=False))
        scoreKM.append(100 * ML_KMeans(df2, 'Grade'))
        scoreKNN.append(100 * ML_KNN(df2, 'Grade', neighbors=5, p=1))
    print('scoreRF: \n', scoreRF)
    print('scoreDT: \n', scoreDT)
    print('scoreKM: \n', scoreKM)
    print('scoreKNN: \n', scoreKNN)
    score_list = [scoreRF, scoreDT, scoreKM, scoreKNN]
    score_labels = ['Random Forest', 'Decision Tree', 'KMeans', 'KNeighborsClassifier']
    for scores, label in zip(score_list, score_labels):
        plt.plot(scores, label=label)
        plt.xlabel('Run Time(count)')
        plt.ylabel('Score(%)')
    plt.legend()
    plt.title('Train Scores after Normalization')
    plt.show()

    # MLP
    print('Train Scores before Normalization')
    AI_MLP_Keras(df, 'Grade', hidden_layers=3)
    print('Train Scores after Normalization')
    AI_MLP_Keras(df2, 'Grade', hidden_layers=3)
