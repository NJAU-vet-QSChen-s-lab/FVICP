# =============================================================================
# PyTorch数据输出脚本
# =============================================================================
# 功能：为深度学习准备数据，按细胞类型导出表达矩阵

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

# 读取Seurat对象
fascia <- qread(file.path(OUTPUT_DIR, "seurat", "fascia.qs"))
table(fascia$orig.ident)

#------------------独立输出每一个细胞类型的二分类训练矩阵-----------------------
setwd(PROJECT_ROOT)
dir.create(file.path(OUTPUT_DIR, "pytorch"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(OUTPUT_DIR, "pytorch", "train"), showWarnings = FALSE, recursive = TRUE)
setwd(file.path(OUTPUT_DIR, "pytorch", "train"))

scelist <- SplitObject(fascia, split.by = "celltype")
namelist <- names(scelist)
for (j in 1:length(scelist)) {
  samplesclist <- SplitObject(scelist[[j]], split.by = "sample")
  # 创建一个空的数据框用于存储所有样本的数据
  merged_data <- NULL
  # 对每个样本类型进行循环
  samples <- unique(scelist[[j]]$sample)
  for (sample in samples) {
    # 获取当前样本和细胞类型的数据
    data <- data.frame(samplesclist[[sample]]@assays$RNA@data)
    # 修改列名
    colnames(data) <- rep(paste(sample, sep = "_"), ncol(data))
    # 合并数据
    if (is.null(merged_data)) {
      merged_data <- data
    } else {
      merged_data <- cbind(merged_data, data)
      head(merged_data, c(3L, 3L))
    }
  }
  unique(colnames(merged_data))
  length(unique(colnames(merged_data)))
  # 生成文件名
  filename <- paste0("fascia", "_", namelist[[j]], ".csv")
  # 输出数据到CSV文件
  write.csv(merged_data, file = filename, row.names = FALSE)
}

#------------------独立输出每一个细胞类型的二分类测试矩阵-----------------------
setwd(PROJECT_ROOT)
dir.create(file.path(OUTPUT_DIR, "pytorch"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(OUTPUT_DIR, "pytorch", "test"), showWarnings = FALSE, recursive = TRUE)
setwd(file.path(OUTPUT_DIR, "pytorch", "test"))

scelist <- SplitObject(fascia, split.by = "celltype")
namelist <- names(scelist)
for (j in 1:length(scelist)) {
  samplesclist <- SplitObject(scelist[[j]], split.by = "sample")
  # 创建一个空的数据框用于存储所有样本的数据
  merged_data <- NULL
  # 对每个样本类型进行循环
  samples <- unique(scelist[[j]]$sample)
  for (sample in samples) {
    # 获取当前样本和细胞类型的数据
    data <- data.frame(samplesclist[[sample]]@assays$RNA@data)
    # 修改列名
    colnames(data) <- rep(paste(sample, sep = "_"), ncol(data))
    # 合并数据
    if (is.null(merged_data)) {
      merged_data <- data
    } else {
      merged_data <- cbind(merged_data, data)
      head(merged_data, c(3L, 3L))
    }
  }
  unique(colnames(merged_data))
  length(unique(colnames(merged_data)))
  # 生成文件名
  filename <- paste0("fascia", "_", namelist[[j]], ".csv")
  # 输出数据到CSV文件
  write.csv(merged_data, file = filename, row.names = TRUE)
}

#---------------------独立输出整个筋膜的二分类测试矩阵--------------------------
setwd(PROJECT_ROOT)
dir.create(file.path(OUTPUT_DIR, "pytorch"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(OUTPUT_DIR, "pytorch", "test"), showWarnings = FALSE, recursive = TRUE)
setwd(file.path(OUTPUT_DIR, "pytorch", "test"))

samplesclist <- SplitObject(fascia, split.by = "sample")
# 创建一个空的数据框用于存储所有样本的数据
merged_data <- NULL
# 对每个样本类型进行循环
samples <- unique(fascia$sample)
for (sample in samples) {
  # 获取当前样本和细胞类型的数据
  data <- data.frame(samplesclist[[sample]]@assays$RNA@data)
  # 修改列名
  colnames(data) <- rep(paste(sample, sep = "_"), ncol(data))
  # 合并数据
  if (is.null(merged_data)) {
    merged_data <- data
  } else {
    merged_data <- cbind(merged_data, data)
    head(merged_data, c(3L, 3L))
  }
}
unique(colnames(merged_data))
length(unique(colnames(merged_data)))
# 生成文件名
filename <- paste0("fascia", "_all_test", ".csv")
# 输出数据到CSV文件
write.csv(merged_data, file = filename, row.names = TRUE)

cat("=== PyTorch数据输出完成 ===\n")
cat("训练数据已保存到:", file.path(OUTPUT_DIR, "pytorch", "train"), "\n")
cat("测试数据已保存到:", file.path(OUTPUT_DIR, "pytorch", "test"), "\n")
