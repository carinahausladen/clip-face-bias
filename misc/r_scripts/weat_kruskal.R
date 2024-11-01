rm(list=ls())
library(dplyr)


causalface <- read_csv("../embs/results_causalface.csv")
fairface <- read_csv("../embs/results_fairface.csv")
utkface <- read_csv("../embs/results_utkface.csv")


perform_kruskal_test <- function(diff_value) {
  
  temp_function <- function(df, diff_value, which_df_name) {
    df %>%
      filter(diff == diff_value) %>%
      mutate(which_df = which_df_name)
  }
  
  causalface_temp <- temp_function(causalface, diff_value, 'causalface')
  fairface_temp <- temp_function(fairface, diff_value, 'fairface')
  utkface_temp <- temp_function(utkface, diff_value, 'utkface')
  
  combined_df <- rbind(causalface_temp, fairface_temp, utkface_temp)
  kruskal.test(eff_size ~ which_df, data = combined_df)
}


kruskal_results_male_female <- perform_kruskal_test("male_female")
kruskal_results_asian_black <- perform_kruskal_test("asian_black")
kruskal_results_white_black <- perform_kruskal_test("white_black")
kruskal_results_asian_white <- perform_kruskal_test("asian_white")


kruskal_results_male_female
kruskal_results_asian_black
kruskal_results_white_black
kruskal_results_asian_white
