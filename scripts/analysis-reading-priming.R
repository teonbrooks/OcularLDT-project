library("languageR")
library("lme4")
library("ez")
library("ggplot2")
control = lmerControl(optimizer='bobyqa')


data <- read.csv("/Volumes/teon-backup/Experiments/E-MEG/data/group/group_OLDT_fixation_times.txt")
data = data[!is.na(data$ffd),]
data = data[data$word == 'True',]
data$subject = factor(data$subject)

# remove outliers
ffd.mean = mean(data$ffd, na.rm=TRUE)
ffd.std = sd(data$ffd, na.rm=TRUE)
data[data$ffd > ffd.mean + 3*ffd.std | data$ffd < ffd.mean - 3*ffd.std,] = NA

# MLM
model <- lmer(ffd~priming + (1+priming|subject), data = data)
means = aggregate(ffd ~ priming, data = data, FUN = mean)
std = aggregate(ffd ~ priming, data = data, FUN = sd)
