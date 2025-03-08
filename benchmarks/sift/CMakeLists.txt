add_executable(
    load_sift
    load_sift.cc utils.cc utils.h
)

target_link_libraries(load_sift PRIVATE ${PROJECT_NAME} FAISS::faiss) 

add_executable(
    search_sift
    search_sift.cc utils.cc utils.h
)

target_link_libraries(search_sift PRIVATE ${PROJECT_NAME} FAISS::faiss) 
