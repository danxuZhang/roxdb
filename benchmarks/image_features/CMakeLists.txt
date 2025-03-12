add_executable(
    roxdb_add
    roxdb_add.cc utils.cc utils.h io.cc io.h query.h
)

target_link_libraries(roxdb_add PRIVATE ${PROJECT_NAME} FAISS::faiss ${HDF5_LIBRARIES}) 

add_executable(
    roxdb_search
    roxdb_search.cc utils.cc utils.h io.h io.cc query.h
)

target_link_libraries(roxdb_search PRIVATE ${PROJECT_NAME} FAISS::faiss ${HDF5_LIBRARIES}) 
if(OpenMP_CXX_FOUND)
    target_link_libraries(roxdb_search PUBLIC OpenMP::OpenMP_CXX)
    target_compile_options(roxdb_search PRIVATE -DWITH_OPENMP -fmarch=native -O3 -fopenmp)
endif()

add_executable(
    roxdb_add_search
    roxdb_add_search.cc utils.cc utils.h io.h io.cc query.h
)

target_link_libraries(roxdb_add_search PRIVATE ${PROJECT_NAME} FAISS::faiss ${HDF5_LIBRARIES}) 

