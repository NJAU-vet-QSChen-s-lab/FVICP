library(cellcall)
Cell_call <- function(object, cell_type, Org = "Mus musculus"){
  library(cellcall)
  test <- CreateObject_fromSeurat(Seurat.object= object, #seurat对象
                                  slot="counts", 
                                  cell_type=cell_type, #细胞类型
                                  data_source="UMI",
                                  scale.factor = 10^6, 
                                  Org = Org) #物种信息
  mt <- TransCommuProfile(object = test,
                          pValueCor = 0.05,
                          CorValue = 0.1,
                          topTargetCor=1,
                          p.adjust = 0.05,
                          use.type="median",
                          probs = 0.9,
                          method="mean",
                          IS_core = TRUE,
                          Org = Org)
  return(mt)
}

