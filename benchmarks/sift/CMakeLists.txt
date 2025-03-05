add_executable(
    single_vector 
    single_vector.cc utils.cc utils.h
)

target_link_libraries(single_vector PRIVATE ${PROJECT_NAME} FAISS::faiss) 

add_executable(
    multi_vector 
    multi_vector.cc utils.cc utils.h
)

target_link_libraries(multi_vector PRIVATE ${PROJECT_NAME} FAISS::faiss) 
