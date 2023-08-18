install.packages(tidyverse)
library(tidyverse)
theme_set(theme_light())
install.packages("reticulate")
library(reticulate)
install.packages("tensorflow")
library(tensorflow)
install_tensorflow(extra_packages="pillow")
install.packages("keras")
library(keras)
py_install("scipy")
py_available("spicy")


#Image Process function
process_image <- function(images) {
  img <- lapply(images, image_load, grayscale = TRUE) # grayscale the image
  arr <- lapply(img, image_to_array) # turns it into an array
  arr_resized <- lapply(arr, image_array_resize, 
                        height = 100, 
                        width = 100) # resize
  arr_normalized <- normalize(arr_resized, axis = 1) #normalize to make small numbers 
  return(arr_normalized)
}

# Load imges with covid
images <- list.files("C:/Users/alexa/OneDrive/Υπολογιστής/Covid Ct scans/COVID", full.names = TRUE) 
covid <- process_image(images)
covid <- covid[,,,1] # get rid of last dim
covid_reshaped <- array_reshape(covid, c(nrow(covid), 100*100))
# Load images without covid
images <- list.files("C:/Users/alexa/OneDrive/Υπολογιστής/Covid Ct scans/non-COVID", full.names = TRUE) 
non_covid <- process_image(images)
non_covid <- non_covid[,,,1] # get rid of last dim
non_covid_reshaped <- array_reshape(non_covid, c(nrow(non_covid), 100*100))

#Visulise random convid and noncovid pic
scancovid <- reshape2::melt(covid[10,,])
plotcovid <- scancovid %>%
  ggplot() +
  aes(x = Var1, y = Var2, fill = value) + 
  geom_raster() +
  labs(x = NULL, y = NULL, title = "CT scan of a patient with covid") + 
  scale_fill_viridis_c() + 
  theme(legend.position = "none")

scannon_covid <- reshape2::melt(non_covid[10,,])
plotnon_covid <- scannon_covid %>%
  ggplot() +
  aes(x = Var1, y = Var2, fill = value) + 
  geom_raster() +
  labs(x = NULL, y = NULL, title = "CT scan of a patient without covid") + 
  scale_fill_viridis_c() + 
  theme(legend.position = "none")

library(patchwork)
plotcovid + plotnon_covid

#Combine the matrixes

df <- rbind(cbind(covid_reshaped, 1), # 1 = covid
            cbind(non_covid_reshaped, 0)) # 0 = no covid
set.seed(100)
#Shuffle observations in df
shuffle <- sample(nrow(df), replace = F)
df <- df[shuffle, ]
# Split df into training and test subsets
set.seed(200)
split <- sample(2, nrow(df), replace = T, prob = c(0.8, 0.2))
train <- df[split == 1,]
test <- df[split == 2,]
train_target <- df[split == 1, 10001] # label in training dataset
test_target <- df[split == 2, 10001] # label in testing dataset

#Build a CNN model
model <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu") %>% 
  layer_dropout(0.4) %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 2, activation = 'softmax')



# Complile the model for binary classification
model %>%
  compile(optimizer = 'adam',
          loss = 'binary_crossentropy', 
          metrics = c('accuracy'))


# Dummy code train and test target
train_label <- to_categorical(train_target)
test_label <- to_categorical(test_target)
# Fit the cnn model to train
fit_covid <- model %>%
  fit(x = train,
      y = train_label, 
      epochs = 30,
      batch_size = 512, # Can also try 128 and 256
      verbose = 2,
      validation_split = 0.2)
plot(fit_covid)



#Predictions on test 
model %>%
  evaluate(test, test_label)



pred <- model %>% predict(test)
pred <- round(pred)
# Confusion matrix
confusion_matrix <- table(Predicted_Values=pred,Actual_Values=test_label)
confusion_matrix


#Load new sets of images for evaluation
# Load imges with covid
images <- list.files("C:/Users/alexa/OneDrive/Υπολογιστής/Covid Ct scans/CT_COVID", full.names = TRUE) 
covid2 <- process_image(images)
covid2 <- covid2[,,,1] # get rid of last dim
covid_reshaped2 <- array_reshape(covid2, c(nrow(covid2), 100*100))
# Load images without covid
images <- list.files("C:/Users/alexa/OneDrive/Υπολογιστής/Covid Ct scans/CT_NonCOVID", full.names = TRUE) 
non_covid2 <- process_image(images)
non_covid2 <- non_covid2[,,,1] # get rid of last dim
non_covid_reshaped2 <- array_reshape(non_covid2, c(nrow(non_covid2), 100*100))

#Combine the matrixes

df2 <- rbind(cbind(covid_reshaped2, 1), # 1 = covid
            cbind(non_covid_reshaped2, 0)) # 0 = no covid
set.seed(100)
#Shuffle observations in df
shuffle2 <- sample(nrow(df2), replace = F)
df2 <- df2[shuffle2, ]
test_target2 <- df2[, 10001]
test_label2 <- to_categorical(test_target2)

model %>%
  evaluate(df2, test_label2)


pred2 <- model %>% predict(df2)
pred2 <- round(pred2)
# Confusion matrix
confusion_matrix <- table(Predicted_Values=pred2,Actual_Values=test_label2)
confusion_matrix

head.matrix(covid_reshaped)

git clone https://github.com/Alex-Stavrop/covid-scans-classification.git]

