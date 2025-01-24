depth = -1
max_flash = 2 * 1024 * 1024
max_sram = 264 * 1024 

flash_needed = 0
sram_needed = 0

while (True):
    depth = depth + 1
    flash_needed = (2**(depth-1)) * (5 * depth + 1)
    if (flash_needed >= max_flash):
        depth = depth - 1
        flash_needed = (2**(depth-1)) * (5 * depth + 1)
        break

print(flash_needed, "/", max_flash, ", depth: ", depth)

depth = -1

while (True):
    depth = depth + 1
    nodes = 0
    for i in range(depth+1):
        nodes = nodes + 2**i
    sram_needed = 16 * nodes
    if (sram_needed >= max_sram):
        depth = depth - 1
        nodes = 0
        for i in range(depth+1):
            nodes = nodes + 2**i
        sram_needed = 16 * nodes
        break

print(sram_needed, "/", max_sram, ", depth: ", depth)