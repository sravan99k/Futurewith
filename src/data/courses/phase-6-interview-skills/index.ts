import { Phase, TopicFolder } from '../types';

const folders: TopicFolder[] = [
    {
        "id": "folder-6-01-technical_interview_strategies",
        "name": "01-technical_interview_strategies",
        "description": "Topics for 01-technical_interview_strategies",
        "topics": [
            {
                "id": "6-01-0",
                "title": "Technical Interview Strategies Theory",
                "description": "Learning about 01_technical_interview_strategies_theory",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-6-interview-skills/01-technical_interview_strategies/01_technical_interview_strategies_theory.md"
            },
            {
                "id": "6-01-1",
                "title": "Technical Interview Strategies Practice Exercises",
                "description": "Learning about 02_technical_interview_strategies_practice_exercises",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-6-interview-skills/01-technical_interview_strategies/02_technical_interview_strategies_practice_exercises.md"
            },
            {
                "id": "6-01-2",
                "title": "Technical Interview Strategies Cheatsheet",
                "description": "Learning about 03_technical_interview_strategies_cheatsheet",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-6-interview-skills/01-technical_interview_strategies/03_technical_interview_strategies_cheatsheet.md"
            }
        ]
    },
    {
        "id": "folder-6-02-coding_interview_patterns",
        "name": "02-coding_interview_patterns",
        "description": "Topics for 02-coding_interview_patterns",
        "topics": [
            {
                "id": "6-02-0",
                "title": "Coding Interview Patterns Theory",
                "description": "Learning about 01_coding_interview_patterns_theory",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-6-interview-skills/02-coding_interview_patterns/01_coding_interview_patterns_theory.md"
            },
            {
                "id": "6-02-1",
                "title": "Coding Interview Patterns Practice Exercises",
                "description": "Learning about 02_coding_interview_patterns_practice_exercises",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-6-interview-skills/02-coding_interview_patterns/02_coding_interview_patterns_practice_exercises.md"
            },
            {
                "id": "6-02-2",
                "title": "Coding Interview Patterns Cheatsheet",
                "description": "Learning about 03_coding_interview_patterns_cheatsheet",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-6-interview-skills/02-coding_interview_patterns/03_coding_interview_patterns_cheatsheet.md"
            }
        ]
    },
    {
        "id": "folder-6-03-system_design_interviews",
        "name": "03-system_design_interviews",
        "description": "Topics for 03-system_design_interviews",
        "topics": [
            {
                "id": "6-03-0",
                "title": "System Design Interviews Theory",
                "description": "Learning about 01_system_design_interviews_theory",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-6-interview-skills/03-system_design_interviews/01_system_design_interviews_theory.md"
            },
            {
                "id": "6-03-1",
                "title": "System Design Interviews Practice Exercises",
                "description": "Learning about 02_system_design_interviews_practice_exercises",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-6-interview-skills/03-system_design_interviews/02_system_design_interviews_practice_exercises.md"
            },
            {
                "id": "6-03-2",
                "title": "System Design Interviews Cheatsheet",
                "description": "Learning about 03_system_design_interviews_cheatsheet",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-6-interview-skills/03-system_design_interviews/03_system_design_interviews_cheatsheet.md"
            }
        ]
    },
    {
        "id": "folder-6-04-behavioral_interview_frameworks",
        "name": "04-behavioral_interview_frameworks",
        "description": "Topics for 04-behavioral_interview_frameworks",
        "topics": [
            {
                "id": "6-04-0",
                "title": "Behavioral Interview Frameworks Theory",
                "description": "Learning about 01_behavioral_interview_frameworks_theory",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-6-interview-skills/04-behavioral_interview_frameworks/01_behavioral_interview_frameworks_theory.md"
            },
            {
                "id": "6-04-1",
                "title": "Behavioral Interview Frameworks Practice Exercises",
                "description": "Learning about 02_behavioral_interview_frameworks_practice_exercises",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-6-interview-skills/04-behavioral_interview_frameworks/02_behavioral_interview_frameworks_practice_exercises.md"
            },
            {
                "id": "6-04-2",
                "title": "Behavioral Interview Frameworks Cheatsheet",
                "description": "Learning about 03_behavioral_interview_frameworks_cheatsheet",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-6-interview-skills/04-behavioral_interview_frameworks/03_behavioral_interview_frameworks_cheatsheet.md"
            }
        ]
    },
    {
        "id": "folder-6-05-company_specific_preparation",
        "name": "05-company_specific_preparation",
        "description": "Topics for 05-company_specific_preparation",
        "topics": [
            {
                "id": "6-05-0",
                "title": "Company Specific Preparation Theory",
                "description": "Learning about 01_company_specific_preparation_theory",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-6-interview-skills/05-company_specific_preparation/01_company_specific_preparation_theory.md"
            },
            {
                "id": "6-05-1",
                "title": "Company Specific Preparation Practice Exercises",
                "description": "Learning about 02_company_specific_preparation_practice_exercises",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-6-interview-skills/05-company_specific_preparation/02_company_specific_preparation_practice_exercises.md"
            },
            {
                "id": "6-05-2",
                "title": "Company Specific Preparation Cheatsheet",
                "description": "Learning about 03_company_specific_preparation_cheatsheet",
                "duration": "1-2 hours",
                "type": "theory",
                "markdownPath": "/courses/phase-6-interview-skills/05-company_specific_preparation/03_company_specific_preparation_cheatsheet.md"
            }
        ]
    }
];

const allTopics = folders.flatMap(folder => folder.topics);

export const phase6: Phase = {
  id: 'phase-6',
  number: 6,
  title: 'Interview Skills',
  description: 'Master technical and behavioral interview strategies',
  duration: '4-6 weeks',
  skills: ["Interview Prep","Coding","System Design","Behavioral"],
  topics: allTopics,
  folders: folders
};
