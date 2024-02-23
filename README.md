# EST_Classifier

**Burokratt-Classifier-Prototype-Rootcode**

This repository serves as a prototype for the Burokratt-Classifier. The following sections provide details:

## 1. How to Run the Prototype

Ensure that Docker and Git are properly installed and configured. Then, clone the GitHub repository using the following command:

```bash
git clone https://github.com/rootcodelabs/Burokratt-Classifier-Prototype-Rootcode.git
```

Navigate to the cloned repository directory.

Build and run the Docker containers using the following Docker command:

```bash
docker-compose up --build
```

And then you can view the web app by going to [http://localhost:3000/](http://localhost:3000/)

To view logs, use the command:

```bash
docker-compose logs -f
```

To stop the containers, use the command:

```bash
docker-compose down
```

After running the Docker containers, access the web application using the provided web address.

## 2. What is Included in the Prototype

The prototype consists of the following main sections accessible from the sidebar:

1. **Dataset**:
   - View current datasets and import new datasets in JSON or CSV format.
   - Expectations for unlabeled JSON files: `["Example1....","Example2....","Example3....",....]`
   - Expectations for labeled JSON files:
     ```json
     {
         "class1":["Example1....","Example2....","Example3....",....],
         "class2":["Example4....","Example5....","Example6....",....],
         "class3":["Example7....","Example8....","Example9....",....]
     }
     ```
   - Expectations for unlabeled CSV files: CSV file with one column containing example data for a class.

2. **Classes**:
   - Add, view, or delete classes.
   - Add more data to existing classes.

3. **Models**:
   - View accuracy, F1 score, and precision of existing models for each class.
   - Start creating a model by selecting a class and base NLP model (Bert, Albert, and XLNet).
   - Note: Models are trained for only one epoch due to prototype limitations.

4. **Test**:
   - Test a trained model by selecting the model and passing string input.

## 3. Special Points to Consider

- YAML files and labeled CSV files are not considered in the prototype.
- All models are set to train for one epoch to save time and resources, but this is configurable through code.
