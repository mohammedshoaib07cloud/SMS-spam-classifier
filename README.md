# SMS-spam-classifier
ML project to classify SMS messages as spam or ham

1.first we imported all the importent librarys into our notebook

2. then we imported our dataset
3. got the requred info on the dataset
4. cleaned the dataset , droped the duplicate vlaues
5. split the data and applyed  1.MultinomialNB modal,2.Logistic Regression ,3.Logistic Regression + Extra featurs
6. fetched the 1.Accuracy, 2.Precision, 3.Recall
7. compered the outputs of the three modals which is shown blow:
8. Model	                         Accuracy	Precision	Recall	   F1  
 0	TF-IDF + Naive Bayes	           0.966      0.990	     0.740	  0.847
 1	TF-IDF + Logistic Regression	   0.976	  0.921	     0.885	  0.903
 2	+ extra features	               0.979	  0.916	     0.916	  0.916

From the above table i can chose modal 1,3
  1. modal 1 beause it detects more ham and less spam which is good caues if it detected any real email as spam that is worse then allowing       span to it
  2. 2.modal 3 if it can just shift the real emails into the spam folder that could be searched then it is ok cause it can't detect ham as        ham efficiently like other modal but it porfoms better then the privious modals


What i would improve?
 .Train the modal on bigger dataset 
 .feature scalling
 .