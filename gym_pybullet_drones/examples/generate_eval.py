import csv
import random
import argparse

def generate_csv(n, filename):
    filename = filename + '.csv'
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(['startx', 'y', 'starty', 'goalx', 'goaly'])
        
        for _ in range(n):
            startx = round(random.uniform(-0.3, 0), 2)
            # Generate y between -1.1 and 0.1, rounded to 2 decimals
            y = round(random.uniform(-1.1, 0.1), 2)
            # Generate start between y-0.3 and y+0.3
            starty = random.uniform(y - 0.3, y + 0.3)
            # Compute goal based on the condition for y
            # Generate goalx between 2.7 and 4.2
            goalx = round(random.uniform(2.7, 3.9), 2)
            if y > -0.5:
                goaly = y + 0.5
            else:
                goaly = y - 0.5
            
            writer.writerow([startx, y, starty, goalx, goaly])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a CSV file with random data for y, start, and goal.")
    parser.add_argument("--filename", type=str, help="Name of the CSV file to generate")
    parser.add_argument("--n", type=int, help="Number of lines to generate in the CSV file")
    args = parser.parse_args()
    
    generate_csv(args.n, args.filename)
    print("File " + args.filename + ".csv generated with " + str(args.n) + " lines.")
