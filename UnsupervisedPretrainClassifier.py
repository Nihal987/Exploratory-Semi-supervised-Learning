class UnsupervisedPretrainClassifier():

    def __init__(self,base_model):
        self.base_model = base_model
        self.model = Sequential()
        self.output_layer = None

    def fit(self,X_train,y_train,X_ulab):
        self.model.add(Dense(10, input_dim=X.shape[1], activation='relu', kernel_initializer='he_uniform')) #input_dim = Number of features for X
        self.model.add(Dense(X.shape[1], activation='softmax')) 
        # compile model
        opt = SGD(learning_rate=0.01, momentum=0.9)
        self.model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

        # Combining the Labelled and un unlabelled data
        X = np.append(X_train,X_ulab,axis=0)

        callback = EarlyStopping(monitor='accuracy', patience=5)
        # Fit the model to be able to represent all the features
        self.model.fit(X, X, epochs=100,callbacks=[callback])

        # remember the current output layer
        self.output_layer = self.model.layers[-1]
        self.model.pop()

        # Keeping the Feature representation
        for layer in self.model.layers:
            layer.trainable = False
      
        y_train = to_categorical(y_train)
        # New Output layer
        self.model.add(Dense(2, activation='softmax')) # Output dimension = y dimensions

        self.model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.01, momentum=0.9), metrics=['accuracy'])
        self.model.fit(X_train,y_train,epochs=100,callbacks=[callback])

    def predict(self,X):
        return self.model.predict(X)

    # Function to add another layer and then retrain the model
    def add_layer(self,X,y,unlabel_percent=0.5):
        self.model.pop()
        self.fit(X,y,unlabel_percent=0.5)
