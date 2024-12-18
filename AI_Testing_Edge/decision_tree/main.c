#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DEPTH 3

struct TreeNode{
    float condition;
    int index;
    int result;
    struct TreeNode * below;
    struct TreeNode * over;
};

typedef struct TreeNode TreeNode;
typedef TreeNode * TreeNodeIndex;

TreeNodeIndex buildTree(int * index, float condition[DEPTH][1 << (DEPTH - 1)], int * results, int level, int rowNumber){
    TreeNodeIndex node = (TreeNodeIndex)malloc(sizeof(TreeNode));
    if (!node) {
        return NULL;
    }
    if (level >= DEPTH){
        node->result = results[rowNumber];
        node->below = NULL;
        node->over = NULL;
        return node;
    }
    node->index = index[level];
    node->condition = condition[level][rowNumber];
    node->below = buildTree(index, condition, results, level+1, rowNumber*2);
    node->over = buildTree(index, condition, results, level+1, rowNumber*2+1);
    return node;
}


int main(void){
    int index[DEPTH] = {0,1,2};
    int powerDepth = (int)pow(2, DEPTH-1);
    float condition[DEPTH][1 << (DEPTH - 1)] = {
        {5},
        {6,7},
        {8,9,10,11}
    };
    int results[1 << DEPTH] = {0,1,0,1,0,1,0,1};
    TreeNodeIndex treeInit = buildTree(index, condition, results, 0, 0);
    printf("Tree Built\n");
    printf("TreeNode result: %d\n", treeInit->below->over->below->result);
    return 0;
}