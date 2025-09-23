# =============================================================================
# 细胞通讯分析脚本
# =============================================================================
# 功能：使用CellCall方法识别配体-受体对，分析细胞间通讯模式

print("Hello world!")
rm(list = ls())

# 加载配置文件
source("../config.R")

# 设置工作目录
setwd(ENV_DIR)
lf <- list.files("./")
for (file in lf) {
  source(file)
}
rm(lf, file)

# 设置项目目录
setwd(PROJECT_ROOT)

# 创建细胞通讯分析目录
dir.create(file.path(OUTPUT_DIR, "cellcall"), showWarnings = FALSE, recursive = TRUE)
setwd(file.path(OUTPUT_DIR, "cellcall"))

library(scMetabolism)
library(ggplot2)
library(rsvd)
library(RImagePalette)
library(imager)
library(scales)
library(ggplot2)
library(ggprism)
library(reshape)
library(ggalluvial)
library(plotly)
library(paletteer)

# 读取Seurat对象
fascia <- qread(file.path(OUTPUT_DIR, "seurat", "fascia.qs"))
fascia <- subset(fascia, subset = celltype2 %in% c("Others", "MuSC"), invert = T)
sclist <- SplitObject(fascia, split.by = "sample")
namelist <- names(sclist)
namelist

# 加载所需的包
library(foreach)
library(doParallel)

# 设置并行环境
numCores <- detectCores() # 获取可用的核心数
cl <- makeCluster(numCores) # 创建 cluster
registerDoParallel(cl) # 注册并行后端

# 并行计算: 使用 foreach 实现
cellcalllist <- foreach(sce=sclist, .combine=list, .multicombine=TRUE, .packages = "cellcall") %dopar% {
  Cell_call(sce, "celltype2", Org = "Mus musculus") 
}
saveRDS(cellcalllist, "cellcalllist.rds")
names(cellcalllist) <- namelist
# 停止并行环境
stopCluster(cl)

for (i in 1:length(cellcalllist)) {
  saveRDS(cellcalllist[[i]], paste0("cellcall", namelist[i], ".rds"))
  LR <- cellcalllist[[i]]@data$expr_l_r_log2_scale
  LR <- LR[which(rowSums(LR) > 0),]
  LR <- LR[, which(colSums(LR) > 0)]
  LR <- na.omit(LR)
  write.csv(LR, paste0(namelist[i], "_cellcall_LR.csv"))
}

file_path <- "cellcalllist.rds"
if (file.exists(file_path)) {
  file.remove(file_path)
  message(paste0(file_path, "已被删除。"))
} else {
  message("文件不存在。")
}

cat("=== 细胞通讯分析完成 ===\n") 