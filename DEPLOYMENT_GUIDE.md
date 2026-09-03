# 🚀 Deployment Guide: Marbl (Meat QC Pipeline)

Deploying an AI computer vision project for the first time can seem daunting, but it boils down to two main things:
1. **Containerizing the app** so it runs exactly the same in the cloud as it does on your laptop.
2. **Hosting it on a platform** that automatically provides HTTPS (which is strictly required by mobile browsers to open the camera).

We have already added a `Dockerfile` to the root of this repository, which handles step 1 automatically! 

Here is the complete, beginner-friendly workflow to deploy this project live to the internet using **Render**, one of the easiest modern hosting platforms.

---

## 🛠️ Prerequisites
Before you start, make sure all your latest code is pushed to your GitHub repository.

---

## ☁️ Step 1: Create a Render Account
1. Go to [Render.com](https://render.com/) and sign up for a free account using your GitHub account.
2. Once logged in, you will be taken to your Render Dashboard.

---

## 🚢 Step 2: Deploy as a Web Service
1. Click the **"New +"** button at the top right of the dashboard and select **"Web Service"**.
2. Under "Connect a repository", find and select your `project-meat-analysis` repository. (If you don't see it, click "Configure account" on the right side to give Render permission to see that specific GitHub repo).
3. Fill out the deployment details:
   - **Name:** `marbl-app` (or whatever you like)
   - **Region:** Choose the region closest to you (e.g., Frankfurt or Ohio)
   - **Branch:** `main`
   - **Root Directory:** *(leave blank)*
   - **Runtime:** `Docker` *(Render should automatically detect the `Dockerfile` we just created!)*
4. **Instance Type:** 
   - Select the **Free** tier to start. 
   - *Note: Free tier instances "spin down" after 15 minutes of inactivity, meaning the very first time someone opens the app after a long break, it might take ~50 seconds to boot up. If you need it to be instantly available 24/7 for a professional demo, upgrade to the $7/mo Starter tier.*
5. Scroll down and click **Create Web Service**.

---

## ⏳ Step 3: Wait for the Build
Render is now building a cloud computer from scratch using our `Dockerfile`. You will see a terminal window on your screen showing the progress.

**What is the Dockerfile doing behind the scenes?**
- It downloads a lightweight Linux operating system.
- It installs the C++ libraries required by OpenCV.
- It installs a special CPU-only version of PyTorch (since cloud GPUs are very expensive, we use the CPU version to keep hosting cheap/free. MobileNetV2 is small enough to run in ~100ms on a CPU anyway!).
- It copies over your frontend, backend, and the trained AI model weights.

*This process usually takes 3 to 5 minutes.*

---

## 🎉 Step 4: Access Your Live App
Once the terminal prints `Uvicorn running on http://0.0.0.0:8000` and Render shows a green **"Live"** badge, you are done!

1. Look near the top left of your Render dashboard for a URL that looks like `https://marbl-app.onrender.com`.
2. Click it! 
3. Because Render automatically secures your app with HTTPS, you can now open this link on your phone, and the mobile browser will happily grant access to the camera!

---

## 🔄 Updating Your App in the Future
The best part about this setup is that it is linked directly to your GitHub repository.

If you ever want to update the app (e.g., you retrained the model to be more accurate, or changed the UI colors):
1. Make the changes on your laptop.
2. Commit and push the changes to GitHub (`git push`).
3. Render will detect the push, automatically rebuild the Docker container, and silently swap the live version with your new version without you having to click a single button.
