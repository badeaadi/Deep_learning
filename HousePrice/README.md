# Deep learning



## Bucharest housing dataset


### Dataset Description

In the dataset there are over three thousand apartments listed for sale on the locally popular website imobiliare.ro. Each entry provides details about different aspects of the house or apartment:

- Nr Camere indicates the number of rooms;
- Suprafata specifies the total area of the dwelling;
- Etaj specifies the floor that the home is located at;
- Total Etaje is the total number of floors of the block of flats;
- Sector represents the administrative district of Bucharest in which the apartment is located;
- Pret represents the listing price of each dwelling;
- Scor represents a rating between 1 and 5 of location of the apartment. It was computed in the following manner by the dataset creator:

The initial dataset included the address of each flat;

An extra dataset was used, which included the average sales price of dwellings in different areas of town;

Using all of these monthly averages, a clusterization algorithm grouped them into 5 classes, which were then labelled 1-5;


### Assignment : 

- Predict the Nr Camere of each dwelling, treating it as a classification problem. Choose an appropriate loss function;
- Predict the Nr Camere of each dwelling, treating it as a regression problem. Choose an appropriate loss function;
Compare the results of the two approaches, displaying the Confusion Matrix for the two, as well as any comparing any other metrics you think are interesting (e.g. MSE). Comment on the results;
- Choose to predict a feature more suitable to be treated as a regression problem, then successfully solve it.

