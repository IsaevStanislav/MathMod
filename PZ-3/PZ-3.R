library('ISLR') 
library('GGally') 
library('MASS') 

my.seed <- 123 
train.percent <- 0.75 
options("ggmatrix.progress.bar" = FALSE) 

library('titanic') 
head(titanic_train) 
?titanic


str(titanic_train) 
titanic_train <- titanic_train[,-4] 
titanic_train <- titanic_train[,-8] 
titanic_train <- titanic_train[,-9] 
titanic_train$Survived <- factor(titanic_train$Survived) 
# графики разброса
ggp <- ggpairs(titanic_train) 
print(ggp, progress = FALSE) 

# Logit ========================================================================== 
set.seed(my.seed) 
inTrain <- sample(seq_along(titanic_train$Survived), 
                  nrow(titanic_train)*train.percent) 
df <- titanic_train[inTrain, ] 
# фактические значения на обучающей выборке 
Факт <- df$Survived 
model.logit <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + PassengerId, data = df, family = 'binomial') 
summary(model.logit) 
model.logit <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + PassengerId, data = df, family = 'binomial') 
summary(model.logit) 
model.logit <- glm(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare, data = df, family = 'binomial') 
summary(model.logit) 
model.logit <- glm(Survived ~ Pclass + Sex + Age + SibSp +Fare, data = df, family = 'binomial') 
summary(model.logit) 
model.logit <- glm(Survived ~ Pclass + Sex + Age + SibSp, data = df, family = 'binomial') 
summary(model.logit) 

# прогноз: вероятности принадлежности классу '1' (выжил) 
p.logit <- predict(model.logit, df, type = 'response') 
Прогноз <- factor(ifelse(p.logit > 0.5, 2, 1), 
                  levels = c(1, 2), 
                  labels = c('0', '1')) 
# матрица неточностей 
conf.m <- table(Факт, Прогноз) 
conf.m 

# чувствительность 
conf.m[2, 2] / sum(conf.m[2, ]) 
# специфичность 
conf.m[1, 1] / sum(conf.m[1, ]) 
# верность 
sum(diag(conf.m)) / sum(conf.m) 

#LDA=============================================================

model.lda <- lda(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + PassengerId, data = df) 
model.lda 

# прогноз: вероятности принадлежности классу 'Yes' (дефолт) 
p.lda <- predict(model.lda, df, 
                 type = 'response') 
Прогноз <- factor(ifelse(p.lda$posterior[, '1'] > 0.5, 
                         2, 1), 
                  levels = c(1, 2), 
                  labels = c('0', '1')) 

# матрица неточностей 
conf.m <- table(Факт, Прогноз) 
conf.m 

# чувствительность 
conf.m[2, 2] / sum(conf.m[2, ]) 
# специфичность 
conf.m[1, 1] / sum(conf.m[1, ]) 
# верность 
sum(diag(conf.m)) / sum(conf.m) 
#ROC

# ROC-кривая =========================================================== 
png(filename = 'Roc-кривая на обучающей.png', bg = 'white') 
# считаем 1-SPC и TPR для всех вариантов границы отсечения 
x <- NULL # для (1 - SPC) 
y <- NULL # для TPR 
x1 <- NULL 
y1 <- NULL 
# заготовка под матрицу неточностей 
tbl <- as.data.frame(matrix(rep(0, 4), 2, 2)) 
rownames(tbl) <- c('fact.No', 'fact.Yes') 
colnames(tbl) <- c('predict.No', 'predict.Yes') 
# вектор вероятностей для перебора 
p.vector <- seq(0, 1, length = 501) 
# цикл по вероятностям отсечения 
for (p in p.vector){ 
  # прогноз 
  Прогноз <- factor(ifelse(p.lda$posterior[, '1'] > p, 
                           2, 1), 
                    levels = c(1, 2), 
                    labels = c('0', '1')) 
  
  # фрейм со сравнением факта и прогноза 
  df.compare <- data.frame(Факт = Факт, Прогноз = Прогноз) 
  
  # заполняем матрицу неточностей 
  tbl[1, 1] <- nrow(df.compare[df.compare$Факт == '0' & df.compare$Прогноз == '0', ]) 
  tbl[2, 2] <- nrow(df.compare[df.compare$Факт == '1' & df.compare$Прогноз == '1', ]) 
  tbl[1, 2] <- nrow(df.compare[df.compare$Факт == '0' & df.compare$Прогноз == '1', ]) 
  tbl[2, 1] <- nrow(df.compare[df.compare$Факт == '1' & df.compare$Прогноз == '0', ]) 
  
  # считаем характеристики 
  TPR <- tbl[2, 2] / sum(tbl[2, 2] + tbl[2, 1]) 
  y <- c(y, TPR) 
  SPC <- tbl[1, 1] / sum(tbl[1, 1] + tbl[1, 2]) 
  x <- c(x, 1 - SPC) 
} 
for (p in p.vector){ 
  # прогноз 
  Прогноз1 <- factor(ifelse(p.logit > p, 
                            2, 1), 
                     levels = c(1, 2), 
                     labels = c('0', '1')) 
  
  # фрейм со сравнением факта и прогноза 
  df.compare <- data.frame(Факт = Факт, Прогноз1 = Прогноз1) 
  
  #заполняем матрицу неточностей 
  tbl[1, 1] <- nrow(df.compare[df.compare$Факт == '0' & df.compare$Прогноз == '0', ]) 
  tbl[2, 2] <- nrow(df.compare[df.compare$Факт == '1' & df.compare$Прогноз == '1', ]) 
  tbl[1, 2] <- nrow(df.compare[df.compare$Факт == '0' & df.compare$Прогноз == '1', ])
  tbl[2, 1] <- nrow(df.compare[df.compare$Факт == '1' & df.compare$Прогноз == '0', ]) 
  
  # считаем характеристики 
  TPR <- tbl[2, 2] / sum(tbl[2, 2] + tbl[2, 1]) 
  y1 <- c(y1, TPR) 
  SPC <- tbl[1, 1] / sum(tbl[1, 1] + tbl[1, 2]) 
  x1 <- c(x1, 1 - SPC) 
} 
# строим ROC-кривую 
par(mar = c(5, 5, 1, 1)) 
# кривая 
plot(x, y, type = 'l', col = 'blue', lwd = 1, 
     xlab = '(1 - SPC)', ylab = 'TPR', 
     xlim = c(0, 1), ylim = c(0, 1)) 
lines(x1, y1, type = 'l', col = 'red', lwd = 1, 
      xlab = '(1 - SPC)', ylab = 'TPR', 
      xlim = c(0, 1), ylim = c(0, 1)) 
# прямая случайного классификатора 
abline(a = 0, b = 1, lty = 3, lwd = 2) 
# точка для вероятности 0.5 
points(x[p.vector == 0.5], y[p.vector == 0.5], pch = 16) 
text(x[p.vector == 0.5], y[p.vector == 0.5], 'p = 0.5(lda)', pos = 4) 
# точка для вероятности 0.2 
points(x[p.vector == 0.2], y[p.vector == 0.2], pch = 16) 
text(x[p.vector == 0.2], y[p.vector == 0.2], 'p = 0.2(lda)', pos = 4) 

# точка для вероятности 0.5 
points(x1[p.vector == 0.5], y1[p.vector == 0.5], pch = 16) 
text(x1[p.vector == 0.5], y1[p.vector == 0.5], 'p = 0.5(logit)', pos = 4) 
# точка для вероятности 0.2 
points(x1[p.vector == 0.2], y1[p.vector == 0.2], pch = 16) 
text(x1[p.vector == 0.2], y1[p.vector == 0.2], 'p = 0.2(logit)', pos = 4) 
dev.off()

df1 <- titanic_train[-inTrain, ] 
# фактические значения на обучающей выборке 
Факт1 <- df1$Survived 
# Logit2 ========================================================================== 

model.logit <- glm(Survived ~ Pclass + Sex + Age + SibSp, data = df1, family = 'binomial') 
summary(model.logit) 

p.logit <- predict(model.logit, df1, type = 'response') 
Прогноз <- factor(ifelse(p.logit > 0.5, 2, 1), 
                  levels = c(1, 2), 
                  labels = c('0', '1')) 

# LDA2 ========================================================================== 
model.lda <- lda(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare, data = df1)
model.lda 
# прогноз: вероятности принадлежности классу 'выжил' 
p.lda <- predict(model.lda, df1, type = 'response') 
Прогноз <- factor(ifelse(p.lda$posterior[, '1'] > 0.5, 
                         2, 1), 
                  levels = c(1, 2), 
                  labels = c('0', '1')) 


# считаем 1-SPC и TPR для всех вариантов границы отсечения 
x <- NULL # для (1 - SPC) 
y <- NULL # для TPR 
x1 <- NULL 
y1 <- NULL 
# заготовка под матрицу неточностей 
tbl <- as.data.frame(matrix(rep(0, 4), 2, 2)) 
rownames(tbl) <- c('fact.No', 'fact.Yes') 
colnames(tbl) <- c('predict.No', 'predict.Yes') 
# вектор вероятностей для перебора 
p.vector <- seq(0, 1, length = 501) 
# цикл по вероятностям отсечения 
for (p in p.vector){ 
  # прогноз 
  Прогноз <- factor(ifelse(p.lda$posterior[, '1'] > p, 
                           2, 1), 
                    levels = c(1, 2), 
                    labels = c('0', '1')) 
  
  # фрейм со сравнением факта и прогноза 
  df.compare <- data.frame(Факт1 = Факт1, Прогноз = Прогноз) 
  
  # заполняем матрицу неточностей 
  tbl[1, 1] <- nrow(df.compare[df.compare$Факт == '0' & df.compare$Прогноз == '0', ]) 
  tbl[2, 2] <- nrow(df.compare[df.compare$Факт == '1' & df.compare$Прогноз == '1', ]) 
  tbl[1, 2] <- nrow(df.compare[df.compare$Факт == '0' & df.compare$Прогноз == '1', ]) 
  tbl[2, 1] <- nrow(df.compare[df.compare$Факт == '1' & df.compare$Прогноз == '0', ]) 
  
  # считаем характеристики 
  TPR <- tbl[2, 2] / sum(tbl[2, 2] + tbl[2, 1]) 
  y <- c(y, TPR) 
  SPC <- tbl[1, 1] / sum(tbl[1, 1] + tbl[1, 2]) 
  x <- c(x, 1 - SPC) 
} 
for (p in p.vector){ 
  # прогноз 
  Прогноз1 <- factor(ifelse(p.logit > p, 
                            2, 1), 
                     levels = c(1, 2), 
                     labels = c('0', '1')) 
  
  # фрейм со сравнением факта и прогноза 
  df.compare <- data.frame(Факт1 = Факт1, Прогноз1 = Прогноз1) 
  
  #заполняем матрицу неточностей 
  tbl[1, 1] <- nrow(df.compare[df.compare$Факт == '0' & df.compare$Прогноз == '0', ]) 
  tbl[2, 2] <- nrow(df.compare[df.compare$Факт == '1' & df.compare$Прогноз == '1', ]) 
  tbl[1, 2] <- nrow(df.compare[df.compare$Факт == '0' & df.compare$Прогноз == '1', ]) 
  tbl[2, 1] <- nrow(df.compare[df.compare$Факт == '1' & df.compare$Прогноз == '0', ]) 
  
  # считаем характеристики 
  TPR <- tbl[2, 2] / sum(tbl[2, 2] + tbl[2, 1]) 
  y1 <- c(y1, TPR) 
  SPC <- tbl[1, 1] /
    sum(tbl[1, 1] + tbl[1, 2]) 
  x1 <- c(x1, 1 - SPC) 
} 
# строим ROC-кривую 
par(mar = c(5, 5, 1, 1)) 
# кривая 
plot(x, y, type = 'l', col = 'blue', lwd = 2, #qda 
     xlab = '(1 - SPC)', ylab = 'TPR', 
     xlim = c(0, 1), ylim = c(0, 1)) 
lines(x1, y1, type = 'l', col = 'red', lwd = 2, #logit 
      xlab = '(1 - SPC)', ylab = 'TPR', 
      xlim = c(0, 1), ylim = c(0, 1)) 
# прямая случайного классификатора 
abline(a = 0, b = 1, lty = 3, lwd = 2) 

# точка для вероятности 0.5 
points(x[p.vector == 0.5], y[p.vector == 0.5], pch = 16) 
text(x[p.vector == 0.5], y[p.vector == 0.5], 'p = 0.5(lda)', pos = 4) 
# точка для вероятности 0.2 
points(x[p.vector == 0.2], y[p.vector == 0.2], pch = 16) 
text(x[p.vector == 0.2], y[p.vector == 0.2], 'p = 0.2(lda)', pos = 4) 

# точка для вероятности 0.5 
points(x1[p.vector == 0.5], y1[p.vector == 0.5], pch = 16) 
text(x1[p.vector == 0.5], y1[p.vector == 0.5], 'p = 0.5(logit)', pos = 4) 
# точка для вероятности 0.2 
points(x1[p.vector == 0.2], y1[p.vector == 0.2], pch = 16) 
text(x1[p.vector == 0.2], y1[p.vector == 0.2], 'p = 0.2(logit)', pos = 4)


#Сравнивая модели по Roc-кривым на обучающей выборке можно сделать вывод о том, что модель LDA лучше с задачей выявления выживших, чем модель логистической регрессии. 
#При уменьшениии вероятности с 0.5 до 0.2  у обеих моделей увеличивается чувствителньость,однако так же увеличивается и доля ложных срабатываний. 
#Причем, доля ложных срабатываний у модели логистической регрессии больше, чем у модели LDA.
#Сравнивая модели по Roc-кривым на тестовой выборке можно сделать аналогичные выводы.
#Что повлияло на преимущество метода - ответить сложно, так как Несмотря на различия, логистическая регрессия и линейный дискриминантный анализ дают схожие результаты.
