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

add_executable(
    roxdb_add_search
    roxdb_add_search.cc utils.cc utils.h io.h io.cc query.h
)

target_link_libraries(roxdb_add_search PRIVATE ${PROJECT_NAME} FAISS::faiss ${HDF5_LIBRARIES}) 

