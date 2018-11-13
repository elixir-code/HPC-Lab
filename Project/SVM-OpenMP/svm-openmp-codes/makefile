CXX = g++
LD = g++

# path #
SRC_PATH = src
BUILD_PATH = build

TEST_PATH = test
BIN_PATH = $(BUILD_PATH)/bin

# code lists #
SOURCES = $(wildcard $(SRC_PATH)/*.cpp)
OBJECTS = $(SOURCES:$(SRC_PATH)/%.cpp=$(BUILD_PATH)/%.o)
DEPENDENCIES = $(OBJECTS:%.o=%.d)

TEST_SOURCES = $(wildcard $(TEST_PATH)/*.cpp)
TEST_OBJECTS = $(TEST_SOURCES:$(TEST_PATH)/%.cpp=$(BUILD_PATH)/%.o)
TEST_DEPENDENCIES = $(TEST_OBJECTS:%.o=%.d)

BINARIES = $(TEST_SOURCES:$(TEST_PATH)/%.cpp=$(BIN_PATH)/%)

# flags #
INCLUDES = -I include

CXXFLAGS = -Wall -std=c++11 -fopenmp
LDFLAGS = -fopenmp



.PHONY: directories
all: directories program

-include $(DEPENDENCIES) $(TEST_DEPENDENCIES)

# ----------------------------------------------------- #

directories: $(BUILD_PATH) $(BIN_PATH)

$(BUILD_PATH):
	mkdir -p  $(BUILD_PATH)

$(BIN_PATH):
	mkdir -p $(BIN_PATH)	

# ----------------------------------------------------- #

.PHONY: program
program: $(BINARIES)

$(BIN_PATH)/%: $(OBJECTS) $(BUILD_PATH)/%.o
	$(LD) $(LDFLAGS) $^ -o $@

$(BUILD_PATH)/%.o: $(SRC_PATH)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -c $< -o $@

$(BUILD_PATH)/%.o: $(TEST_PATH)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -MMD -c $< -o $@

# ----------------------------------------------------- #

.PHONY: clean
clean:
	rm -rf $(BIN_PATH)
	rm -rf $(BUILD_PATH)

# ----------------------------------------------------- #

.SECONDARY: $(OBJECTS) $(TEST_OBJECTS)

# ----------------------------------------------------- #