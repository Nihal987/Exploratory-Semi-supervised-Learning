class Ensemble():
    def __init__(self,dt,rf,svm,knn,gb):
        dt = dt
        rf = rf
        svm = svm
        knn = knn
        gb = gb
        self.ensemble = VotingClassifier(estimators=[('dt', dt), ('rf', rf), ('svm', svm), ('knn',knn),('gb',gb)], voting='hard')

    def fit(self,X_lab,y_lab,X_ulab,split_percent=0.25):
        if(X_ulab is None):
            self.ensemble.fit(X_lab,y_lab)
            return
        while len(X_ulab)>4:

            self.ensemble.fit(X_lab,y_lab)
            y_pred = self.ensemble.predict(X_lab)

            X_ulab,y_ulab, X_pseudo,y_pseudo = train_test_split(X_ulab,y_pred,test_size=split_percent,random_state=42)

            X_lab = np.append(X_lab,X_pseudo,axis=0)
            y_lab = np.append(y_lab,y_pseudo)

    def predict(self,X):
        return self.ensemble.predict(X)