library("languageR")
library("lme4")
library("ggplot2")
control = lmerControl(optimizer='bobyqa')


data <- read.csv("/Volumes/teon-backup/Experiments/E-MEG/data/group/group_OLDT_region_times.txt")
data = data[!is.na(data$dur),]
data = data[data$word == 1,]
data$subject = factor(data$subject)
data$dur = as.numeric(data$dur)
data$trial = as.numeric(data$trial)


# remove outliers
data = data[data$dur > 80,]
dur.mean = mean(data$dur, na.rm=TRUE)
dur.std = sd(data$dur, na.rm=TRUE)
data[data$dur > dur.mean + 3*dur.std | data$dur < dur.mean - 3*dur.std,] = NA

# select target region
target = data[data$ia == 'target',]
# prime = data[data$ia == 'prime',]


# MLM
model <- lmer(dur~priming + (1+priming|subject), data = target)
means = aggregate(dur ~ priming, data = target, FUN = mean)
std = aggregate(dur ~ priming, data = target, FUN = sd)

# MLM
# model <- lmer(dur~priming + (1+priming|subject), data = prime)
# means = aggregate(dur ~ priming, data = prime, FUN = mean)

cat("\014") 
summary(model)
means
std
