#include "data.h"

#include <stdio.h>
#include <stdlib.h>
#include "pico/stdlib.h"
#include "pico/time.h"

#define DEPTH 6
#define LED_PIN 14

struct TreeNode{
    float condition;
    int index;
    int result;
    int level;
    int rowNumber;
    struct TreeNode * below;
    struct TreeNode * over;
};

typedef struct TreeNode TreeNode;
typedef TreeNode * TreeNodeIndex;

TreeNodeIndex buildTree(int index[DEPTH][1 << (DEPTH-1)], float condition[DEPTH][1 << (DEPTH-1)], int * results, int level, int rowNumber){
    TreeNodeIndex node = (TreeNodeIndex)malloc(sizeof(TreeNode));
    if (!node) {
        return NULL;
    }
    node -> level = level;
    node -> rowNumber = rowNumber;
    if (level >= DEPTH){
        node->result = results[rowNumber];
        node->below = NULL;
        node->over = NULL;
        return node;
    }
    node->index = index[level][rowNumber];
    node->condition = condition[level][rowNumber];
    node->below = buildTree(index, condition, results, level+1, rowNumber*2);
    node->over = buildTree(index, condition, results, level+1, rowNumber*2 + 1);
    return node;
}

int getResult(float * data, TreeNodeIndex initTree){
    TreeNodeIndex node = initTree;
    while(node->below!=NULL){
        if(data[node->index]>node->condition){
            node = node->over;
        }
        else{
            node = node->below;
        }
    }
    return node->result;
}

int main() {
    stdio_init_all();
    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);
    sleep_ms(2000);
    int index[DEPTH][1 << (DEPTH-1)] =  {
        {10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,1,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,3,9,10,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,9,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,7,0,0},
    };
    float condition[DEPTH][1 << (DEPTH-1)] =   {
        {0.026835039258003235,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,20.882240295410156,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,9.5,23.94119167327881,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,21.209446907043457,0.048063503578305244,0.05069274269044399,5.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0.07197032496333122,54.67359733581543,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
        {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,61.73231315612793,0,0},
    };
    int results[1 << (DEPTH)] = {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,0,0,0,0,1,1,1,1,3,3,3,3,2,2,2,2,1,1,1,1,0,0,1,1,0,0,1,0,0,0,0,0};
    TreeNodeIndex TREE = buildTree(index, condition, results, 0, 0);
    printf("Tree built\n");

    int result;
    int current = 0;
    while (true) {
        result = getResult(data_array[current], TREE);
        printf("Result: %d\n", result);
        if (current >= DATA_ROWS) current = 0;
        else current++;
        gpio_put(LED_PIN, 1);
        sleep_ms((result+1)*250);
        gpio_put(LED_PIN, 0);
        sleep_ms(5*1000);
    }
    return 0;
}