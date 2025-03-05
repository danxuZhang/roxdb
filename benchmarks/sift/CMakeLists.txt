add_executable(
    single_vector 
    single_vector.cc utils.cc utils.h
)

target_link_libraries(single_vector PRIVATE ${PROJECT_NAME} FAISS::faiss) 
