# FIXME(Ying): This may lead to runtime error if users have multiple locally
# installed Pythons. To avoid runtime error, it is better to explicitly specify
# which python is in use through: cmake -DPYTHON_EXECUTABLE:FILEPATH=`which
# python3`

find_package(PythonLibs REQUIRED)

message(STATUS "Python include dir: ${PYTHON_INCLUDE_DIR}")
message(STATUS "Python library: ${PYTHON_LIBRARY}")

add_library(python SHARED IMPORTED GLOBAL)
set_property(TARGET python PROPERTY IMPORTED_LOCATION ${PYTHON_LIBRARIES})
include_directories(${PYTHON_INCLUDE_DIRS})
