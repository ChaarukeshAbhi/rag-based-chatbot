# This is Instruction file which contains instructions on how to build this project using any AI tool.

## About the project
This project is a RAG based Enterprise HR Assistant Platform separated features based on roles. This fits naturally with Role-Based Access Control (RBAC) and must make the project feel like a real product.

### Employee Features

Employees are mostly consumers of information and service requests.
They will have
Profile & Dashboard

*   Must be able to view their profile details
*   View department/team that they are working
*   Their Employee ID under The profile
*   Can update personal information
*   View new Announcements
*   Raise tickets
*   Apply for a leave
*   Check status of their leave permissions
*   check for pending leaves.
*   Also a chatbot to get informations like what is updated HR policies ?

Other Features List given below
| Feature             | Employee | HR | Admin |
| ------------------- | -------: | -: | ----: |
| Chat with AI        |        ✅ |  ✅ |     ✅ |
| View policies       |        ✅ |  ✅ |     ✅ |
| Apply leave         |        ✅ |  ✅ |     ✅ |
| Approve leave       |        ❌ |  ✅ |     ✅ |
| View analytics      |        ❌ |  ✅ |     ✅ |
| Manage employees    |        ❌ |  ✅ |     ✅ |
| Manage system roles |        ❌ |  ❌ |     ✅ |


## Technical Constraints
Use Structure-aware chunking (Chunking using headers like H2 , H3)
Show citations at the end of the answer at right side of answer box UI
Implement LLM Guardrails
check for prompt injection
Analyze the sentiment of the user query. Whether it is a normal question or it is a serious one like grievence , complaint , emergency leave , emergency help. For such cases set the ticket priority to critical. Sentiment detection and ticket priority is explained in below section

Must use ReRanking - Use BM25 and if any content is updated , the retrieval , embedding systems must again chunk (only the necessary part) and again perform reranking
## Sentiments & Ticket Priority
Analyse the user prompt and tell the mood of question Like - Angry , sad , normal , emergency. 
Set the priorities as respective to it

1. Casual
2. Critical
3. Immediate Attention - Classify whether Emergency Healthcare , Harrasement or infrastructure issue.

## Tools and Technologies to be used
FRONTEND 
1. React
2. TailWind CSS
3. TypeScript 
4. Framer Motion
4. Lottie JSON GIFs

BACKEND
1. Fast API
2. Python

SECURITY
1. JWT
2. Password Hashing by bycrypt
3. LLM Guardrails

### Frontend Instructions
1. No Sloggy and usual UI/UX shall be used
2. Proper colour contrast and User must able to toggle between light and dark mode
3. Interactive buttons
4. Typing effect of answers like ChatGPT
5. Greet the user at starting .. (Not just Hey user ! ... But with their name.. must get it from backend)
6. Have a Great UI/UX style ... Not like any fancy Style.


NOTE - NO DEAD CODE OR DEAD FILES MUST BE CREATED AND NO CODE MUST BE WRITTEN UNECESSARLY AND CAUSE LAG OR BUGS. ALL THE CODE PRESENT IN VARIOUS PAGES MUST RUN !
