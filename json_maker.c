#include<stdlib.h>
#include<stdio.h>
#include<stdbool.h>
#include<string.h>


// Function to trim leading and trailing whitespace from a string
char *trim(char *str) {
    while (*str && (*str == ' ' || *str == '\t' || *str == '\n'))
        str++;
    if (*str == '\0')
        return str;
    char *end = str + strlen(str) - 1;
    while (end > str && (*end == ' ' || *end == '\t' || *end == '\n'))
        end--;
    *(end + 1) = '\0';
    return str;
}

int main() {
    FILE *file;
    char line[500]; // Adjust the size according to your needs
    int promptCount = 0;
    bool isPromptSection = false;

    file = fopen("metrics.ipynb", "r");

    if (file == NULL) {
        printf("Error opening file.\n");
        return 1;
    }

    while (fgets(line, sizeof(line), file)) {
        // Check for the beginning of a prompt section
        if (strstr(line, "## prompt") != NULL) {
            isPromptSection = true;
            continue;
        }

        // Check for the end of a prompt section
        if (isPromptSection && strstr(line, "```") != NULL) {
            isPromptSection = false;

            char filename[20]; // Adjust the size according to your needs
            sprintf(filename, "prompt_%d.txt", promptCount);

            FILE *promptFile = fopen(filename, "w");
            if (promptFile == NULL) {
                printf("Error creating prompt file.\n");
                return 1;
            }

            // Write prompt text to the file
            while (fgets(line, sizeof(line), file) && strstr(line, "```") == NULL) {
                // Remove leading and trailing whitespace
                char *trimmedLine = trim(line);
                fputs(trimmedLine, promptFile);
            }

            fclose(promptFile);
            promptCount++;
        }
    }

    fclose(file);
    printf("Prompts extracted successfully.\n");

    return 0;
}

/*
int main() {
    FILE *file;
    char line[500]; // taille arbitraire
    int promptCount = 0;

    file = fopen("metrics.ipynb", "r");

    if (file == NULL) {
        printf("Error opening file.\n");
        return 1;
    }

    while (fgets(line, sizeof(line), file)) {
        if (strstr(line, "## prompt") != NULL) {
            char filename[20]; // taile arbitraire
            sprintf(filename, "prompt_%d.txt", promptCount);

            FILE *promptFile = fopen(filename, "w");
            if (promptFile == NULL) {
                printf("Error creating prompt file.\n");
                return 1;
            }

            // Move to the next line after "## prompt"
            fgets(line, sizeof(line), file);

            // Write prompt text to the file
            while (fgets(line, sizeof(line), file) && strstr(line, "```") == NULL) {
                fputs(line, promptFile);
            }

            fclose(promptFile);
            promptCount++;
        }
    }

    fclose(file);
    printf("Prompts extracted successfully.\n");

    return 0;
}
*/
/*
int num_al = 256;

void notebook_to_json(char* notebook_name, char* json_name){
	FILE* note = fopen(notebook_name,"r");
	//FILE* json = fopen(json_name,"w");
	int i = 0;

	char** prompt;
	// ARTUNG : CA PEUT BEUGUER ICI PCQ char* =/= char[]
	while(feof(note) != 0){
		//on lit jusqu'a tomber sur prompt
		while((feof(note) != 0)&&(strcmp(*prompt,"prompt") != 0)){
			fscanf(note,"%s",*prompt);
		}
		//on lit le prompt correctement
		
		char* num;
		sprintf(num, "%d", i);
		char* name = strncat(json_name,strncat("_",num,num_al),num_al);
		FILE* json = fopen(name,"w");
		while((feof(note) != 0)&&(strcmp(*prompt,"\n") != 0)){
			fscanf(note,"%s",*prompt);
			if(feof(note) != 0){
				fprintf(json,"%s",*prompt);
			}
			if(strcmp(*prompt,"\n") == 0){
				fscanf(note,"%s",*prompt);
				if(feof(note) != 0){
					fprintf(json,"%s",*prompt);
				}
			}
		}
		fclose(json);
	}
	fclose(note);
	//fclose(json)
}

int main(int argc, char** argv){
	notebook_to_json(argv[1],argv[2]);

	return 0;
}*/
