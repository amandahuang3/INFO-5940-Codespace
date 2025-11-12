## Reflection

### What I learn
Actual implementation of a multi-agent workflow gave me a deeper understanding of how different agents can coordinate to complete a task collaboratively. Compared to just listening to the lecture and seeing a live demo on how the multi-agent system works. I learned that having a reviewer agent is definitely helpful for fact-checking, and how each agent specializes in specific tasks can improve their accuracy. As previously, I did try to ask the planner to do both the plan and the review by itself to try to see how good it can get. In general, knowning how to provide agent the tool to use, how to provide helpful agent prompt instruction, log on start & end of tool usage. 

### Challenges Faced
One challenge I face during the implementation is that I got an authentication error on the API key. Though I thought it would be straightforward, just like how we build our RAG application. But as in the previous assignment, I directly provide the API key when I run the command. This time, how the code is set up requires me to put it in the .env file. By having trouble shutting this helps me better understand environment management and API authentication. 

Another challenge was designing the reviewer’s prompt instruction. When I initially set up the instruction, it was just generating the “Delta List” instead of updating what the planner agent had provided. While I want to balance the control and not have the reviewer just update. So, at the bottom, I ask it to provide a summary of the change so users can validate and kind of understand what other information is thought through and what is overlooked.

### Creative/ Variation/ Design Choices
One creative or variation choice is how I instructed the reviewer to internally produce a Delta List of corrections, but only output the reviewed itinerary with a summary change. This design choice can improve the user experience, where users get a polished output, but the model still does the reasoning explicitly in the background to improve its output accuracy. Some design choices I made include ask the planner to include the "address" of the resturant/ activites which can reduce the workload for user to search up again. After the reviwer agent validate the planner suggestion it will explictly say in the title as "Reviewd" for clearity to user that this is validate by the reviwer. 

### External Resource
I reference Microsoft's - [Write Effective Instructions For Declarative Agents](https://learn.microsoft.com/en-us/microsoft-365-copilot/extensibility/declarative-agent-instructions) to help me get started with the structure of the prompt instruction and what is an important component for the agent. 



