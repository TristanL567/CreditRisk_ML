DataPreprocessing <- function(data,Tolerance = 2,
                              ...){
  
  Data_preprocess <- Data %>%
    filter(
      f10 >= -Tolerance,
      (f2 + f3) <= (f1 + Tolerance),
      (f4 + f5) <= (f3 + Tolerance),
      (f6 + f11) <= (f1 + Tolerance)
    )
  return(Data_preprocess)
}