/* Use for HH files with format activity="begin", activity=end" */
/* Code from Dr. Cook */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAXBUFFER 256
#define MAXSTR 80

void Init();
void PrintStats();
void ReadData(FILE *fp);
int FindActivity(char *name);
int ComputeYear(char *date);
int ComputeMonth(char *date);
int ComputeDay(char *date);
int ComputeDow(char *date);
int ComputeHour(char *time);
int ComputeSecond(char *time);

char **activitynames;
int numactivities;
int ddensity[7], hdensity[24], awaydtime[7];
int *adensity, **addensity, **ahdensity;
float *atime, **adtime;

int main(int argc, char *argv[])
{
   FILE *fp;

   fp = fopen(argv[1], "r");
   if (fp == NULL)
   {
      printf("File %s invalid\n", argv[1]);
      exit(1);
   }
   ReadData(fp);
   fclose(fp);
}


void ReadData(FILE *fp)
{
   char *cptr, buffer[MAXBUFFER], temp[MAXSTR];
   char date[MAXSTR], time[MAXSTR], sensorid[MAXSTR], sensorvalue[MAXSTR];
   char status[MAXSTR], alabel[MAXSTR], templabel[MAXSTR];
   int num, length, cactivity=-1;

   cptr = fgets(buffer, 256, fp);
   while (cptr != NULL)
   {
      strcpy(alabel, "none");
      length = strlen(buffer);
             // Ignore lines that are empty or commented lines starting with "%"
      if ((length > 0) && (buffer[0] != '%'))
      {
         while ((length > 1) &&
                ((buffer[length-2] == ' ') || (buffer[length-1] == '	')))
	    length--;
         sscanf(buffer, "%s %s %s %s %s",
	    date, time, sensorid, sensorvalue, alabel);

	 if (strcmp(sensorid, "system") != 0) // Ignore system commands
	 {
            if (strcmp(alabel, "none") != 0)       // There is an activity label
	    {
	       length = strlen(alabel);
               if ((alabel[length-6] == 'b') && (alabel[length-5] == 'e') &&
                    (alabel[length-4] == 'g') && (alabel[length-3] == 'i') &&
                    (alabel[length-2] == 'n') && (alabel[length-1] == '"'))
	       {
	          strncpy(templabel, alabel, length-8);
	          templabel[length-8] = '\0';
	          strcpy(alabel, templabel);
	          num = FindActivity(alabel);
	          if (cactivity == -1)
	             cactivity = num;
	       }
	       else if ((alabel[length-4] == 'e') && (alabel[length-3] == 'n') &&
                        (alabel[length-2] == 'd') && (alabel[length-1] == '"'))
	       {
	          strncpy(templabel, alabel, length-6);
	          templabel[length-6] = '\0';
	          strcpy(alabel, templabel);
	          num = FindActivity(alabel);
	          if (num == cactivity)
	             cactivity = -1;
	       }
               else num = FindActivity(alabel);
	    }
	    else
	    {
	       if (cactivity != -1)
	          strcpy(alabel, activitynames[num]);
	       else strcpy(alabel, "Other_Activity");
	    }
	    printf("%s %s %s %s %s\n", date, time, sensorid, sensorvalue, alabel);
         }
      }

      cptr = fgets(buffer, 256, fp);                           // Get next event
   }
}

// Return index that corresponds to the activity label.  If the label is not
// found in the list of predefined activity labels then a new entry is
// created corresponding to the new label.
int FindActivity(char *name)
{
   int i;

   for (i=0; i<numactivities; i++)
      if (strcmp(name, activitynames[i]) == 0)
         return(i);

   numactivities++;
   if (numactivities == 1)
      activitynames = (char **) malloc(sizeof(char *));
   else
      activitynames = (char **) realloc(activitynames,
                                        numactivities * sizeof(char *));
   activitynames[numactivities-1] = (char *) malloc(MAXSTR * sizeof(char *));
   strcpy(activitynames[numactivities-1], name);
   return(numactivities-1);
}
