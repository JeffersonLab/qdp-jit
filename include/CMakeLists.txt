set( PETE_HDRS PETE/Combiners.h
             PETE/Functors.h
             PETE/PETE.h
             PETE/TreeNodes.h
             PETE/CreateLeaf.h
             PETE/ForEach.h
             PETE/ForEachInOrder.h
	     PETE/ForEachInOrderStatic.h
             PETE/OperatorTags.h
             PETE/Scalar.h
             PETE/TypeComputations.h)
             
install(FILES ${PETE_HDRS} DESTINATION include/PETE)

set( MEMORY_HDRS qdp_allocator.h
  qdp_singleton.h
  qdp_default_allocator.h 
  qdp_qcdoc_allocator.h)

install(FILES ${MEMORY_HDRS} DESTINATION include)

set( CUDA_HDRS qdp_gpu.h
  qdp_cache.h
  qdp_quda.h
  qdp_mapresource.h
  qdp_pool_allocator.h
  qdp_cuda_allocator.h
  qdp_llvm.h
  qdp_viewleaf.h
  qdp_word.h
  qdp_wordjit.h
  qdp_wordreg.h
  qdp_jitfunction.h
  qdp_jitf_op.h
  qdp_jitf_extract.h
  qdp_jitf_spinloop.h
  qdp_jitf_eval.h
  qdp_jitf_subtype.h
  qdp_jitf_latsca.h
  qdp_jitf_pokesite.h
  qdp_jitf_zerorep.h
  qdp_jitf_random.h
  qdp_jitf_gather.h
  qdp_jit_util.h
  qdp_pete_visitors.h
  qdp_qdpsubtypejit.h
  qdp_outerjit.h
  qdp_realityjit.h
  qdp_realityreg.h
  qdp_primscalarjit.h
  qdp_primscalarreg.h
  qdp_basejit.h
  qdp_basereg.h
  qdp_primmatrixjit.h
  qdp_primcolormatjit.h
  qdp_primspinmatjit.h
  qdp_primmatrixreg.h
  qdp_primcolormatreg.h
  qdp_primspinmatreg.h 
  qdp_primseedjit.h
  qdp_primseedreg.h
  qdp_primvectorjit.h
  qdp_primspinvecjit.h
  qdp_primcolorvecjit.h 
  qdp_primvectorreg.h
  qdp_primspinvecreg.h
  qdp_primcolorvecreg.h
  qdp_handle.h
  qdp_mastermap.h
  qdp_masterset.h
  qdp_sum.h
  qdp_datalayout.h
  qdp_jitf_copymask.h
  qdp_jitf_sum.h
  qdp_jitf_summulti.h
  qdp_jitf_globalmax.h
  qdp_jitf_gaussian.h
  qdp_internal.h
  qdp_newopsreg.h
  qdp_jit_function.h
  qdp_expr_writer.h
  qdp_jit_config.h
  )

install(FILES ${CUDA_HDRS} DESTINATION include)

set( QDP_CUSTOM_HDRS
  custom_kernels/custom_kernels.h
  custom_kernels/stream_local_inner_product.h
  )

install(FILES ${QDP_CUSTOM_HDRS} DESTINATION include/)

set( QDP_SPECIAL_HDRS
  qdp_functions_special.h
  cuda_special.h
  )

install(FILES ${QDP_SPECIAL_HDRS} DESTINATION include/)

set( QDP_HDRS
  qdp_config.h
  qdp_forward.h
  qdp_globalfuncs.h
  qdp_globalfuncs_subtype.h
  qdp_init.h
  qdp_io.h
  qdp_stdio.h
  qdp_layout.h
  qdp_map.h
  qdp_multi.h
  qdp_arrays.h
  qdp_newops.h
  qdp_optops.h
  qdp_outer.h
  qdp_outersubtype.h
  qdp_outersubjit.h
  qdp_params.h
  qdp_precision.h
  qdp_primcolormat.h
  qdp_primcolorvec.h
  qdp_primitive.h
  qdp_primmatrix.h
  qdp_primscalar.h
  qdp_primseed.h
  qdp_primspinmat.h
  qdp_primspinvec.h
  qdp_primvector.h
  qdp_profile.h
  qdp_stopwatch.h
  qdp_flopcount.h
  qdp_iogauge.h
  qdp_crc32.h
  qdp_byteorder.h
  qdp_util.h
  qdp_xmlio.h
  qdp_qdpexpr.h
  qdp.h
  qdp_qdpio.h
  QDPOperators.h
  qdp_qdpsubtype.h
  qdp_qdptype.h
  qdp_random.h
  qdp_rannyu.h
  qdp_reality.h
  qdp_simpleword.h
  qdp_specializations.h
  qdp_subset.h
  qdp_traits.h
  qdp_dispatch.h
  qdp_dyn_functions.h
  qdp_functions.h
  qdp_parscalar_specific.h
  qdp_parscalar_mapresource.h
  qdp_parscalar_internal.h
  qdp_scalar_specific.h
  qdp_scalar_internal.h
  qdp_scalar_mapresource.h
  qdp_defs.h
  qdp_scalarsite_defs.h
  qdp_strnlen.h
  qdp_db.h
  qdp_db_imp.h
  qdp_db_stub.h
  qdp_map_obj.h
  qdp_map_obj_null.h
  qdp_map_obj_memory.h
  qdp_map_obj_disk.h
  qdp_map_obj_disk_multiple.h
  qdp_hdf5.h
  qdp_disk_map_slice.h
  )

install( FILES ${QDP_HDRS} DESTINATION include ) 
