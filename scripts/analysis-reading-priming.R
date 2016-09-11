library("languageR")
library("lme4")
library("ez")
library("ggplot2")
control = lmerControl(optimizer='bobyqa')


data <- read.csv("/Volumes/backup/Experiments/E-MEG/data/group/group_OLDT_fixation_times.txt")
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
model.ffd <- lmer(ffd~priming + trial + (1+priming|subject), data = ffd)
means = aggregate(ffd ~ priming, data = ffd, FUN = mean)
std = aggregate(ffd ~ priming, data = ffd, FUN = sd)


# remove outliers
gzd = data
gzd.mean = mean(gzd$dur, na.rm=TRUE)
gzd.std = sd(gzd$dur, na.rm=TRUE)
gzd[gzd$dur > gzd.mean + 3*gzd.std | gzd$dur < gzd.mean - 3*gzd.std,] = NA
gzd = gzd[gzd$ia == 'target',]


# MLM
model.gzd <- lmer(dur~priming + trial + (1+priming|subject), data = data)
means = aggregate(dur ~ priming, data = data, FUN = mean)
std = aggregate(dur ~ priming, data = data, FUN = sd)




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
data = data[data$ia == 'target',]

# MLM
model <- lmer(dur~priming + trial + (1+priming|subject), data = data)
means = aggregate(dur ~ priming, data = data, FUN = mean)
std = aggregate(dur ~ priming, data = data, FUN = sd)

