setwd("C:/Users/szymc/Desktop/eRka")
library(shiny)
library(markdown)

ui <- navbarPage("Database Patients Viewer",
  tabPanel("Import Data",
        
    sidebarLayout(
      sidebarPanel(
          
        helpText("Import CSV files needed to analyse"),
          fileInput("csv", h3("Input Database to analyse")),
          dataframe <- 
          fileInput("template", label = h3("Insert plate template"),
                    accept = c("text/csv", 
                               "text/comma-separated-values,text/plain",
                               ".csv")
    )),
    mainPanel(
      textOutput("selected_csv"),
      textOutput("selected_template"),
      textOutput("csv_toread"),
      textOutput("csv_lenght"),
        )
      )
    ),
  tabPanel("Patient View",
    
    selectInput("patient", label = h3("Choose your patient "), 
              choices = list("Choice 1"=1, "Choice 2"=2, "Choice 3"=3), 
              selected = 1)),
  
  tabPanel("Table view",
           tableOutput("datatable")
  )
)
# Define server logic ----
server <- function(input, output) {
  data <- reactive({
  csv <- input$csv
  if (is.null(csv))
    return(NULL)
  df <- read.csv(csv$datapath, sep = ",",header = TRUE)
  return(df)
  })
  
    output$datatable <- renderTable({data()})
    
    output$csv_lenght <- renderText({
      paste("Number of patients: ", 
            nrow(data()))})
      
    output$selected_csv <- renderText({paste("Elisa reads: ", input$csv[1])})
    output$selected_template <- renderText({paste("Template: ", input$template[1])})
    output$csv_toread <- renderText({paste("Path to file: ", input$datapath)})
    output$dto <- renderDataTable({dataframe()})
  }
shinyApp(ui = ui, server = server)

