library(findPC)
find_PC <- function(sce, scale, nPC = c(10, 20, 30, 40, 50)){
  #data(procdata) #读一个rowname是基因，colname是样本的表达矩阵
  procdata <- as.matrix(sce@assays$RNA@counts)
  sdev<-prcomp(t(procdata),scale. = scale)$sdev[1:50]
  head(sdev)
  
  p <- findPC(sdev = sdev, #需要降序排列
              number = nPC, #输入PC范围，默认为1:30
              method = 'all', #返回所有方法的结果，
              figure = T    #返回结果图片
  )
  return(p)
}

