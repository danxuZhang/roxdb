add_executable(
    roxdb_add
    roxdb_add.cc utils.cc utils.h io.cc io.h
)

target_link_libraries(roxdb_add PRIVATE ${PROJECT_NAME} FAISS::faiss ${HDF5_LIBRARIES}) 

add_executable(
    roxdb_search
    roxdb_search.cc utils.cc utils.h io.h io.cc
)

target_link_libraries(roxdb_search PRIVATE ${PROJECT_NAME} FAISS::faiss ${HDF5_LIBRARIES}) 
