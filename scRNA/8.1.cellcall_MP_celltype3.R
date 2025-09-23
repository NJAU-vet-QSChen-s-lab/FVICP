# =============================================================================
# 巨噬细胞与CPTC细胞通讯分析脚本
# =============================================================================
# 功能：巨噬细胞亚群与CPTC细胞的细胞通讯分析，识别特定细胞类型间的通讯模式

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

# 读取Seurat对象和巨噬细胞数据
fascia <- qread(file.path(OUTPUT_DIR, "seurat", "fascia.qs"))
MP <- qread(file.path(OUTPUT_DIR, "metabolic", "MP.qs"))
fascia <- subset(fascia, subset = celltype2 %in% c("Others", "MuSC"), invert = T)
MP <- subset(MP, subset = celltype2 == "MP")
#-------------------------------------------------------------------------------
CPTC <- subset(fascia, subset = celltype2 == "CPTC")
CPTC$celltype3 <- CPTC$celltype2
test <- merge(MP, y = CPTC)

table(test$group_celltype3)
sclist <- SplitObject(test, split.by = "sample")
namelist <- names(sclist)
for (i in 1:length(sclist)) {
  lt <- length(table(sclist[[i]]$celltype3))
  for (j in lt) {
    sclist[[i]]$group_celltype3 <- paste(namelist[[i]], "_", sclist[[i]]$celltype3, sep = "")
  }
  table(sclist[[i]]$group_celltype3)
}
table(sclist[[1]]$group_celltype3)
table(sclist[[2]]$group_celltype3)
test <- merge(sclist[[1]], y = sclist[[2]])
table(test$group_celltype3)
rm(sclist)

sclist <- SplitObject(test, split.by = "sample")
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
  Cell_call(sce, "celltype3", Org = "Mus musculus") 
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

cat("=== 巨噬细胞-CPTC细胞通讯分析完成 ===\n")
cat("结果已保存到:", file.path(OUTPUT_DIR, "cellcall"), "\n")