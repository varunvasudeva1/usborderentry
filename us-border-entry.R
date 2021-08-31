library(gridExtra)
library(tidyverse)
library(knitr)
library(ggfortify)
library(lindia)
library(leaps)
library(car)
library(GGally)
library(caret)
library(dplyr)

border <- read.csv("Border_Crossing_Entry_Data.csv")
head(border)

border <- border %>%
  mutate(Year = format(as.Date(border$Date, format="%d/%m/%Y"),"%Y"),
         ModYear = case_when(Year >= 1996 & Year <= 2000 ~ "1996 - 2000",
                             Year >= 2001 & Year <= 2005 ~ "2001 - 2005",
                             Year >= 2006 & Year <= 2010 ~ "2006 - 2010",
                             Year >= 2011 & Year <= 2015 ~ "2011 - 2015",
                             Year >= 2016 & Year <= 2020 ~ "2016 - 2020")) %>%
  filter(!Value %in% c(0)) %>%
  drop_na()
head(border)

ggplot(border, aes(x = State, y = Value, col = State)) +
  geom_bar(stat = "identity") +
  facet_wrap(~ModYear) +
  theme(axis.text.x = element_text(angle = 60))

ggplot(border, aes(x = State, y = Value, col = State)) +
  geom_bar(stat = "identity") +
  facet_wrap(~Border) +
  theme(axis.text.x = element_text(angle = 45))

border.full <- lm(Value ~ Port.Name + State + Port.Code + Border + Measure + Year, data = border)
summary(border.full)

border.reduced <- lm(Value ~ Port.Name + Port.Code + Border + Measure + Year, data = border)
summary(border.reduced)

border.backward <- step(border.reduced, direction = "backward")

summary(border.backward)
autoplot(border.backward)

gg_boxcox(border.backward)

border.fit <- lm(log(Value) ~ Port.Name + Measure + Year, data = border)
summary(border.fit)
autoplot(border.fit)

set.seed(536255)
train_control <- trainControl(method = "cv", number = 5)
border.cv <- train(log(Value) ~ Port.Name + Measure + Year, data = border, trControl = train_control, method="lm")
print(border.cv)

new_data = data.frame(Port.Name = "Santa Teresa", Measure = "Buses", Year = "2010")
exp(predict(border.cv, newdata = new_data))

test.df <- data.frame(filter(border, Port.Name == "Santa Teresa" & Measure == "Buses" & Year == "2010"))
mean(test.df$Value)