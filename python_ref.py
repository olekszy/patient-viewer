def testMethod(bins): //get number of bins passed by R Shiny server
  string = "I came from a Python Function" 
  noOfBins = str(bins) //convert int to string to concat
  return "You have selected " +noOfBins+ " bins and " +string
