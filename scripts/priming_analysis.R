library("languageR")
library("lme4")
library("ez")
library("ggplot2")
control = lmerControl(optimizer='bobyqa')


data = read.table("/Volumes/GLYPH-1 TB/Experiments/E-MEG/data/group/group_OLDT_target_times.txt", quote="\"",
                  col.names=c('subject', 'trigger', 'word','prime',
                              'duration', 'mean', 'std'))
data = data[data$word == 1,]
data$subject = factor(data$subject)

# remove outliers
data[data$duration > data$mean + 3*data$std | 
       data$duration < data$mean - 3*data$std,] = NA

# MLM
model <- lmer(duration~prime + (1+prime|subject), data = data)
means = aggregate(duration ~ prime, data = data, FUN = mean)
std = aggregate(duration ~ prime, data = data, FUN = sd)

