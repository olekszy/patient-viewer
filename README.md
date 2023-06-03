# README

This repository contains a Shiny web application for viewing and analyzing patient data from a CSV database. The application allows you to import data, view patient information, and display data in a table format. The application is built using the Shiny package in R.

## Installation

To run the Shiny application, you need to have R installed on your system. You also need to install the following packages:

- shiny
- markdown

You can install these packages by running the following command in R:

```R
install.packages(c("shiny", "markdown"))
```

## Usage

To start the Shiny application, open the R console and run the following command:

```R
library(shiny)
runApp("app.R")
```

This will launch the Shiny application in your web browser.

### Import Data

In the "Import Data" tab, you can upload the CSV file containing the patient database. Click on the "Browse" button and select the CSV file from your local system. You can also upload a plate template CSV file.

### Patient View

In the "Patient View" tab, you can choose a patient from the dropdown menu to view their information. The choices in the dropdown menu are populated based on the data in the imported CSV file.

### Table View

In the "Table View" tab, the imported data is displayed in a table format. You can scroll through the table to view all the patient records. The table is automatically updated when a new CSV file is imported.

## Contributing

If you would like to contribute to this project, you can fork the repository and make your changes. After making the changes, submit a pull request explaining the modifications you have made.

## Issues

If you encounter any issues or bugs while using the application, please open an issue in the repository's issue tracker. Provide a detailed description of the problem, including any error messages or screenshots.

## License

This project is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute the code in this repository.
