library("languageR")
library("lme4")
library("ggplot2")
control = lmerControl(optimizer='bobyqa')


data <- read.csv("/Volumes/backup/Experiments/E-MEG/data/group/group_SENT_fixation_times.txt")
data = data[!is.na(data$ffd),]
data = data[data$word == 1,]
data$subject = factor(data$subject)

# remove implausible fixations
 data = data[data$dur > .080,]

# remove outliers
ffd = data
ffd.mean = mean(ffd$ffd, na.rm=TRUE)
ffd.std = sd(ffd$ffd, na.rm=TRUE)
ffd[ffd$ffd > ffd.mean + 3*ffd.std | ffd$ffd < ffd.mean - 3*ffd.std,] = NA
ffd = ffd[ffd$ia == 'target',]

# MLM
model.ffd <- lmer(ffd~priming + (1+priming|subject), data = ffd)
ffd.means = aggregate(ffd ~ priming, data = ffd, FUN = mean)
ffd.std = aggregate(ffd ~ priming, data = ffd, FUN = sd)


# remove outliers
gzd = data
gzd.mean = mean(gzd$dur, na.rm=TRUE)
gzd.std = sd(gzd$dur, na.rm=TRUE)
gzd[gzd$dur > gzd.mean + 3*gzd.std | gzd$dur < gzd.mean - 3*gzd.std,] = NA
gzd = gzd[gzd$ia == 'target',]


# MLM
model.gzd <- lmer(dur~priming + (1+priming|subject), data = data)
gzd.means = aggregate(dur ~ priming, data = data, FUN = mean)
gzd.std = aggregate(dur ~ priming, data = data, FUN = sd)

cat("\014") 
summary(model.ffd)
ffd.means
ffd.std
summary(model.gzd)
gzd.means
gzd.std
