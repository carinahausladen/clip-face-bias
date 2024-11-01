# tutorial for sc_weat: https://psychbruce.github.io/PsychWordVec/reference/test_WEAT.html

rm(list=ls())
library(PsychWordVec)
library(sweater)
library(dplyr)
library(gt)
library(readr)

#----------------- prep df
text_embs <- read_csv("../embs/text_embs.csv")
unique_embs<-unique(text_embs$attribute)
text_embs$attribute <- paste0(text_embs$attribute, "_", seq_along(text_embs$attribute))

#img_embs <- read_csv("../embs/causalface_img_embs.csv") 
#img_embs <- read_csv("../embs/causalface_ageonly_img_embs.csv") 

img_embs <- read_csv("../embs/fairface_img_embs.csv")
img_embs <- img_embs[,-1]  # somehow first column has rownames

#img_embs <- read_csv("../embs/fairface_img_embs_sampled_1000.csv") 
#img_embs <- read_csv("../embs/fairface_img_embs_sampled_5000.csv")

#img_embs <- read_csv("../embs/utkface_img_embs.csv") 
#img_embs <- read_csv("../embs/utkface_img_embs_sampled_1000.csv") 
#img_embs <- read_csv("../embs/utkface_img_embs_sampled_5000.csv") 


img_embs$gender <- paste0(img_embs$gender, "_", seq_along(img_embs$gender))
img_embs$race <- paste0(img_embs$race, "_", seq_along(img_embs$race))

#----------------- function
calc_WEAT <- function(df_e, A1_label, A2_label, A1_pattern, A2_pattern, attributes) {
  results <- data.frame(
    attribute = character(0),
    mean_diff = numeric(0),
    eff_size = numeric(0),
    p_val = numeric(0)
  )
  
  for (attribute in attributes) {
    T1_pattern <- paste0("^", gsub("\\+", "\\\\+", attribute))
    
    sc_weat <- PsychWordVec::test_WEAT( # calculating single category WEAT
      data = df_e,
      labels = list(T1 = attribute, A1 = A1_label, A2 = A2_label),
      T1 = T1_pattern,
      A1 = A1_pattern,  
      A2 = A2_pattern,  
      use.pattern = TRUE,
      seed = 1
    )
    
    mean_diff <- sc_weat$eff$mean_diff_raw
    eff_size <- sc_weat$eff$eff_size
    p_val <- sc_weat$eff$pval_approx_2sided
    
    results <- rbind(results, data.frame(
      attribute = attribute,
      mean_diff = mean_diff,
      eff_size = eff_size,
      p_val = p_val
    ))
  }
  
  return(results)
}


#----------------- do calc
img_embs2<- img_embs %>% dplyr::select(-c(race, age))  #note that fairface also has age column #'Unnamed: 0'))
colnames(img_embs2)[1]<-"attribute"
df<-rbind(img_embs2,text_embs)
df_e<-as_embed(df,normalize=TRUE)

results_male_female <- calc_WEAT(
  df_e, 
  "Male", "Female", 
  "^male_", "^female_", 
  unique_embs
)

img_embs2<- img_embs %>% dplyr::select(-c(gender, age)) # 'Unnamed: 0'))
colnames(img_embs2)[1]<-"attribute"
df<-rbind(img_embs2,text_embs)
df_e<-as_embed(df,normalize=TRUE)

results_asian_white <- calc_WEAT(
  df_e, 
  "Asian", "White", 
  "^asian_", "^white_", 
  unique_embs
)

results_asian_black <- calc_WEAT(
  df_e, 
  "Asian", "Black", 
  "^asian_", "^black_", 
  unique_embs
)

results_white_black <- calc_WEAT(
  df_e, 
  "White", "Black", 
  "^white_", "^black_", 
  unique_embs
)


results_male_female$diff<-"male_female"
results_asian_black$diff<-"asian_black"
results_white_black$diff<-"white_black"
results_asian_white$diff<-"asian_white"

results_fairface <- rbind(results_male_female,
                 results_asian_black,
                 results_white_black,
                 results_asian_white
                 )

#write.csv(results_fairface, file = "../embs/results_causalface.csv")
#write.csv(results_fairface, file = "../embs/results_causalface_ageonly.csv")

#write.csv(results_fairface, file = "../embs/results_fairface.csv")
#write.csv(results_fairface, file = "../embs/results_fairface_1000.csv")
#write.csv(results_fairface, file = "../embs/results_fairface_5000.csv")

#write.csv(results_fairface, file = "../embs/results_utkface.csv")
#write.csv(results_fairface, file = "../embs/results_utkface_sampled_1000.csv")
#write.csv(results_fairface, file = "../embs/results_utkface_sampled_5000.csv")
  
#34D2EB
  

# Kruskal Wallis test


