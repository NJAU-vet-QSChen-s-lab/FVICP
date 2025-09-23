seurat_process <- function(seurat_object, workers1 = 5, workers2 = 10, maxmem = 32, do_harmony = TRUE){
  library(Seurat)
  #library(monocle3)
  library(tidyverse)
  library(patchwork)
  library(harmony)
  library("parallel")
  
  makecore <- function(workcore, memory){
    if(!require(Seurat)) install.packages('Seurat')
    if(!require(future)) install.packages('future')
    plan("multisession", workers = workcore)
    options(future.globals.maxSize= memory*1024*1024**2)
  }
  
  seurat_object <- NormalizeData(seurat_object, normalization.method = "LogNormalize", scale.factor = 10000)
  cl <- detectCores()
  makecore(workers1, maxmem)
  cl <- makeCluster(workers1)
  seurat_object <- ScaleData(seurat_object, vars.to.regress = c("nCount_RNA", "percent.mt"), verbose = TRUE)
  plan("sequential")
  stopCluster(cl)
  rm(cl)
  
  cl <- detectCores()
  makecore(workers2, maxmem)
  cl <- makeCluster(workers2)
  seurat_object <- FindVariableFeatures(seurat_object, nfeatures = 4000)
  seurat_object <- RunPCA(seurat_object, npcs = 50, verbose = FALSE)
  
  if(do_harmony){
    seurat_object <- RunHarmony(seurat_object, group.by.vars = "orig.ident",
                                assay.use = "SCT", max.iter.harmony = 10)
    seurat_object <- FindNeighbors(seurat_object, reduction = "harmony", dims = 1:50)
    seurat_object <- FindClusters(seurat_object, resolution = seq(from = 0.1, to = 1.0, by = 0.2))
    seurat_object <- RunUMAP(seurat_object, reduction = "harmony", dims = 1:50)
    seurat_object <- RunTSNE(seurat_object, reduction = "harmony", dims = 1:50)
    seurat_object <- JackStraw(object = seurat_object, num.replicate = 100)
    seurat_object <- ScoreJackStraw(object = seurat_object, dims = 1:20)
    
  } else {
    seurat_object <- FindNeighbors(seurat_object, reduction = "pca", dims = 1:50)
    seurat_object <- FindNeighbors(seurat_object, reduction = "pca", dims = 1:50)
    seurat_object <- FindClusters(seurat_object, resolution = seq(from = 0.1, to = 1.0, by = 0.2))
    seurat_object <- RunUMAP(seurat_object, reduction = "pca", dims = 1:50)
    seurat_object <- RunTSNE(seurat_object, reduction = "pca", dims = 1:50)
    seurat_object <- JackStraw(object = seurat_object, num.replicate = 100)
    seurat_object <- ScoreJackStraw(object = seurat_object, dims = 1:20)
    
  }
  
  
  plan("sequential")
  stopCluster(cl)
  rm(cl)
  
  return(seurat_object)
}

maketop10 <- function(seurat_object, object_name, logFCfilter=0.5, adjPvalFilter=0.05){
  seurat_object$seurat_clusters <- seurat_object$RNA_snn_res.0.9
  seurat_object@active.ident <- seurat_object$RNA_snn_res.0.9
  seurat_object$Annot_clusters <- seurat_object$seurat_clusters
  
  plan("multisession", workers = 20) #开启并行计算, future框架会吃掉进度条，别急
  seurat_object.markers <- FindAllMarkers(object = seurat_object,
                                          #group.by = "cellcluster", 
                                          only.pos = FALSE,
                                          min.pct = 0.25,
                                          logfc.threshold = logFCfilter)
  plan("sequential") #停止并行计算
  
  sig.markers <- seurat_object.markers[
    (abs(as.numeric(as.vector(seurat_object.markers$avg_log2FC))) > logFCfilter & 
       as.numeric(as.vector(seurat_object.markers$p_val_adj)) < adjPvalFilter), 
  ]
  
  write.table(sig.markers, file = paste0("04.markers_", object_name, ".xls"), sep = "\t", row.names = F, quote = F)
  
  #绘制阶梯热图
  top10_seurat_object <- sig.markers %>% group_by(cluster) %>% slice_max(n = 10, order_by = avg_log2FC)
  top10_seurat_object <- unique(top10_seurat_object$gene)
  
  return(top10_seurat_object)
}
