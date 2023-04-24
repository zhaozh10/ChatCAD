const chatWindow = document.getElementById("chat-window");
const chatForm = document.getElementById("chat-form");
const userInput = document.getElementById("user-input");

var imageSrc = "none";


// 选择文件后触发事件
document.getElementById('upload-input').addEventListener('change', function(e) {
  var reader = new FileReader();
  reader.onload = function(event) {
    imageSrc = event.target.result; // 获取图片的 base64 编码字符串
    document.getElementById('uploaded-image').src = imageSrc;
    document.getElementById('uploaded-image').style.display = 'block';
  }
  reader.readAsDataURL(e.target.files[0]);
});

chatForm.addEventListener("submit", (event) => {
  event.preventDefault();

  // get user input and clear input field
  message = {text: userInput.value, img: imageSrc};
  userInput.value = "";
  // console.log(JSON.stringify({ message }))

  // add user message to chat window
  const sendbtn = document.querySelector("#submit-input");
  const waitAnimation = document.querySelector("#wait-animation");
  sendbtn.style.display = "none";
  waitAnimation.style.display = "block";
  fetch("/bot", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message }),
  }).then(function (response) {
    const stream = new ReadableStream({
      start(controller) {
        const reader = response.body.getReader();
        function push() {
          reader.read().then(function (result) {
            if (result.done) {
              sendbtn.style.display = "block";
              waitAnimation.style.display = "none";
              controller.close();
              return;
            }
            controller.enqueue(result.value);
            push();
          });
        }
        push();
      },
    });
    const textDecoder = new TextDecoder();
    const readableStreamDefaultReader = stream.getReader();
    let entry_counter = 0;
    function readStream() {
      readableStreamDefaultReader.read().then(function (result) {
        if (result.done) {
          return;
        }

        text = textDecoder.decode(result.value);

        // The server will first send a json containing information about the message just sent
        if (entry_counter == 0) {
          // We parse it and
          infos = JSON.parse(text);
          var temp_message = infos.message;
          temp_message = temp_message.replace(/-/g, "_");
          if (imageSrc !== "none") {
            temp_message = temp_message + "-" + imageSrc;
            imageSrc = "none";
          }

          addMessage("User", temp_message, infos.id, false);
          elements = addMessage("Bot", "", infos.response_id, false);
          messageTextElement = elements["messageTextElement"];
          hiddenElement = elements["hiddenElement"];
          entry_counter++;
          document.getElementById('uploaded-image').style.display = 'none';
        } else {
          // For the other enrtries, these are just the text of the chatbot
          // for (const char of text) {
          //   txt = hiddenElement.innerHTML;
          //   if (char != "\f") {
          //     txt += char;
          //     hiddenElement.innerHTML = txt;
          //     // messageTextElement.innerHTML = txt.replace(/\n/g, "<br>");
          //     messageTextElement.innerHTML = txt;
          //   }

          //   // scroll to bottom of chat window
          //   chatWindow.scrollTop = chatWindow.scrollHeight;
          // }
          hiddenElement.innerHTML = text;
          messageTextElement.innerHTML = text.replace(/\n/g, "<br>");
          chatWindow.scrollTop = chatWindow.scrollHeight;
          entry_counter++;
        }

        readStream();
      });
    }
    readStream();
  });
});

function addMessage(sender, message, id, can_edit = false) {
  // console.log(id);

  if (message.indexOf("data:image") !== -1) {
    var temp_img = message.split("-")[1]
    message = message.split("-")[0]
  }
  else {
    var temp_img = "none"
  }

  const messageElement = document.createElement("div");
  messageElement.classList.add(
    "bg-secondary",
    "drop-shadow-sm",
    "p-4",
    "mx-6",
    "my-4",
    "flex",
    "flex-col",
    "space-x-2"
  );
  messageElement.classList.add(sender);
  messageElement.setAttribute("id", id);

  const senderElement = document.createElement("div");
  senderElement.classList.add("font-normal", "underline", "text-sm");
  senderElement.innerHTML = sender;

  const messageTextElement = document.createElement("div");
  messageTextElement.classList.add("font-medium", "text-md");
  messageTextElement.innerHTML = message;
  // Create a hidden div element needed to buffer responses before commiting them to the visible message
  const hiddenElement = document.createElement("div");
  hiddenElement.style.display = "none";
  hiddenElement.innerHTML = "";

  messageElement.appendChild(senderElement);
  // add img to message
  if (temp_img != "none") {
    const messageImgElement = document.createElement("img");
    messageImgElement.src = temp_img;
    // 创建Canvas元素
    const canvas = document.createElement("canvas");
    const ctx = canvas.getContext("2d");

    // 将图像绘制到Canvas上并缩小五倍
    canvas.width = 250 / messageImgElement.height * messageImgElement.width;
    canvas.height = 250;
    ctx.drawImage(messageImgElement, 0, 0, canvas.width, canvas.height);

    // 将Canvas转换为图像并添加到消息元素中
    const scaledImgElement = new Image();
    scaledImgElement.src = canvas.toDataURL();
    messageTextElement.appendChild(scaledImgElement);
  }
  messageElement.appendChild(messageTextElement);
  if (can_edit) {
    const editButton = document.createElement("button");
    editButton.classList.add(
      "bg-blue-500",
      "hover:bg-blue-700",
      "text-white",
      "font-bold",
      "py-2",
      "px-4",
      "rounded",
      "my-2"
    );
    editButton.innerHTML = "Edit";
    editButton.addEventListener("click", () => {
      const inputField = document.createElement("input");
      inputField.type = "text";
      inputField.classList.add(
        "font-medium",
        "text-md",
        "border",
        "border-gray-300",
        "p-1"
      );
      inputField.value = messageTextElement.innerHTML;

      const saveButton = document.createElement("button");
      saveButton.classList.add(
        "bg-green-500",
        "hover:bg-green-700",
        "text-white",
        "font-bold",
        "py-2",
        "px-4",
        "rounded",
        "my-2",
        "ml-2"
      );
      saveButton.innerHTML = "Save";
      saveButton.addEventListener("click", () => {
        const newText = inputField.value;
        messageTextElement.innerHTML = newText;
        // make request to update message
        const url = `/update_message?id=${id}&message=${newText}`;
        fetch(url)
          .then((response) => {
            if (!response.ok) {
              throw new Error("Network response was not ok");
            } else {
              console.log("Updated");
            }
          })
          .catch((error) => {
            console.error("There was a problem updating the message:", error);
          });
        messageElement.replaceChild(messageTextElement, inputField);
        //messageElement.removeChild(inputField);
        messageElement.removeChild(saveButton);
      });

      messageElement.replaceChild(inputField, messageTextElement);
      messageElement.appendChild(saveButton);
      inputField.focus();
    });

    messageElement.appendChild(editButton);
  }
  chatWindow.appendChild(messageElement);
  chatWindow.appendChild(hiddenElement);

  // scroll to bottom of chat window
  chatWindow.scrollTop = chatWindow.scrollHeight;

  // Return all needed stuff
  return {
    messageTextElement: messageTextElement,
    hiddenElement: hiddenElement,
  };
}

const exportButton = document.getElementById("export-button");

exportButton.addEventListener("click", () => {
  const messages = Array.from(chatWindow.querySelectorAll(".message")).map(
    (messageElement) => {
      const senderElement = messageElement.querySelector(".sender");
      const messageTextElement = messageElement.querySelector(".message-text");
      const sender = senderElement.textContent;
      const messageText = messageTextElement.textContent;
      return { sender, messageText };
    }
  );
  const exportFormat = "json"; // replace with desired export format

  if (exportFormat === "text") {
    const exportText = messages
      .map(({ sender, messageText }) => `${sender}: ${messageText}`)
      .join("\n");
    downloadTextFile(exportText);
  } else if (exportFormat === "json") {
    fetch("/export")
      .then((response) => response.json())
      .then((data) => {
        db_data = JSON.stringify(data);
        // Do something with the data, such as displaying it on the page
        console.log(db_data);
        downloadJsonFile(db_data);
      })
      .catch((error) => {
        // Handle any errors that occur
        console.error(error);
      });
  } else {
    console.error(`Unsupported export format: ${exportFormat}`);
  }
});

function downloadTextFile(text) {
  const blob = new Blob([text], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  downloadUrl(url);
}

function downloadJsonFile(json) {
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  downloadUrl(url);
}

function downloadUrl(url) {
  const link = document.createElement("a");
  link.href = url;
  link.download = "chat.txt";
  link.click();
}

const newDiscussionBtn = document.querySelector("#new-discussion-btn");

newDiscussionBtn.addEventListener("click", () => {
  const discussionName = prompt("Enter a name for the new discussion:");
  if (discussionName) {
    const sendbtn = document.querySelector("#submit-input");
    const waitAnimation = document.querySelector("#wait-animation");
    sendbtn.style.display = "none";
    waitAnimation.style.display = "block";

    // Add the discussion to the discussion list
    const discussionItem = document.createElement("li");
    discussionItem.textContent = discussionName;
    fetch(`/new_discussion?title=${discussionName}`)
      .then((response) => response.json())
      .then((data) => {
        console.log(`New chat ${data}`);
        // Select the new discussion
        //selectDiscussion(discussionId);
        chatWindow.innerHTML = "";
        // addMessage("GPT4ALL", welcome_message, 0);
        addMessage("ChatCAD-plus", welcome_message, 0);
        populate_discussions_list();
        sendbtn.style.display = "block";
        waitAnimation.style.display = "none";
      })
      .catch((error) => {
        // Handle any errors that occur
        console.error(error);
      });
  }
});

function populate_discussions_list() {
  // Populate discussions list
  const discussionsList = document.querySelector("#discussions-list");
  discussionsList.innerHTML = "";
  fetch("/discussions")
    .then((response) => response.json())
    .then((discussions) => {
      discussions.forEach((discussion) => {
        const buttonWrapper = document.createElement("div");
        //buttonWrapper.classList.add('flex', 'space-x-2', 'mt-2');
        buttonWrapper.classList.add(
          "flex",
          "items-center",
          "mt-2",
          "py-4",
          "text-left"
        );

        const renameButton = document.createElement("button");
        renameButton.classList.add(
          "bg-green-500",
          "hover:bg-green-700",
          "text-white",
          "font-bold",
          "py-0",
          "px-0",
          "rounded",
          "mr-2"
        );
        const renameImg = document.createElement("img");
        renameImg.src = "/static/images/edit_discussion.png";
        renameImg.style.width = "20px";
        renameImg.style.height = "20px";
        renameButton.appendChild(renameImg);

        //renameButton.style.backgroundImage = "/rename_discussion.svg"; //.textContent = 'Rename';
        renameButton.addEventListener("click", () => {
          const dialog = document.createElement("dialog");
          dialog.classList.add("bg-white", "rounded", "p-4");

          const inputLabel = document.createElement("label");
          inputLabel.textContent = "New name: ";
          const inputField = document.createElement("input");
          inputField.classList.add(
            "border",
            "border-gray-400",
            "rounded",
            "py-1",
            "px-2"
          );
          inputField.setAttribute("type", "text");
          inputField.setAttribute("name", "title");
          inputField.setAttribute("value", discussion.title);
          inputLabel.appendChild(inputField);
          dialog.appendChild(inputLabel);

          const cancelButton = document.createElement("button");
          cancelButton.textContent = "Cancel";
          cancelButton.addEventListener("click", () => {
            dialog.close();
          });

          const renameConfirmButton = document.createElement("button");
          renameConfirmButton.classList.add(
            "bg-green-500",
            "hover:bg-green-700",
            "text-white",
            "font-bold",
            "py-2",
            "px-4",
            "rounded",
            "ml-2"
          );
          renameConfirmButton.textContent = "Rename";
          renameConfirmButton.addEventListener("click", () => {
            const newTitle = inputField.value;
            if (newTitle === "") {
              alert("New name cannot be empty");
            } else {
              fetch("/rename", {
                method: "POST",
                headers: {
                  "Content-Type": "application/json",
                },
                body: JSON.stringify({ id: discussion.id, title: newTitle }),
              })
                .then((response) => {
                  if (response.ok) {
                    discussion.title = newTitle;
                    discussionButton.textContent = newTitle;
                    dialog.close();
                  } else {
                    alert("Failed to rename discussion");
                  }
                })
                .catch((error) => {
                  console.error("Failed to rename discussion:", error);
                  alert("Failed to rename discussion");
                });
            }
          });

          dialog.appendChild(cancelButton);
          dialog.appendChild(renameConfirmButton);
          document.body.appendChild(dialog);
          dialog.showModal();
        });
        const deleteButton = document.createElement("button");
        deleteButton.classList.add(
          "bg-green-500",
          "hover:bg-green-700",
          "text-white",
          "font-bold",
          "py-0",
          "px-0",
          "rounded",
          "ml-2"
        );
        const deleteImg = document.createElement("img");
        deleteImg.src = "/static/images/delete_discussion.png";
        deleteImg.style.width = "20px";
        deleteImg.style.height = "20px";

        deleteButton.addEventListener("click", () => {
          fetch("/delete_discussion", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ id: discussion.id }),
          })
            .then((response) => {
              if (response.ok) {
                buttonWrapper.remove();
              } else {
                alert("Failed to delete discussion");
              }
            })
            .catch((error) => {
              console.error("Failed to delete discussion:", error);
              alert("Failed to delete discussion");
            });
        });

        deleteButton.appendChild(deleteImg);
        deleteButton.addEventListener("click", () => {});

        const discussionButton = document.createElement("button");
        discussionButton.classList.add(
          "flex-grow",
          "w-full",
          "bg-blue-500",
          "hover:bg-blue-700",
          "text-white",
          "font-bold",
          "py-2",
          "px-4",
          "rounded",
          "text-left",
          "hover:text-white"
        );
        discussionButton.textContent = discussion.title;
        discussionButton.addEventListener("click", () => {
          // send query with discussion id to reveal discussion messages
          fetch("/get_messages", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({ id: discussion.id }),
          })
            .then((response) => {
              if (response.ok) {
                response.text().then((data) => {
                  const messages = JSON.parse(data);
                  console.log(messages);
                  // process messages
                  var container = document.getElementById("chat-window");
                  container.innerHTML = "";
                  messages.forEach((message) => {
                    addMessage(
                      message.sender,
                      message.content,
                      message.id,
                      false
                      // true
                    );
                  });
                });
              } else {
                alert("Failed to query the discussion");
              }
            })
            .catch((error) => {
              console.error("Failed to get messages:", error);
              alert("Failed to get messages");
            });
          console.log(`Showing messages for discussion ${discussion.id}`);
        });

        buttonWrapper.appendChild(renameButton);
        buttonWrapper.appendChild(deleteButton);
        buttonWrapper.appendChild(discussionButton);
        discussionsList.appendChild(buttonWrapper);
      });
    })
    .catch((error) => {
      console.error("Failed to get discussions:", error);
      alert("Failed to get discussions");
    });
}

// First time we populate the discussions list
populate_discussions_list();

function add_collapsible_div(discussion_title, text, id) {
  // Create the outer box element
  const box = document.createElement("div");
  box.classList.add("bg-gray-100", "rounded-lg", "p-4");

  // Create the title element
  const title = document.createElement("h2");
  title.classList.add("text-lg", "font-medium");
  title.textContent = discussion_title;

  // Create the toggle button element
  const toggleBtn = document.createElement("button");
  toggleBtn.classList.add("focus:outline-none");
  toggleBtn.id = `${id}-toggle-btn`;

  // Create the expand icon element
  const expandIcon = document.createElement("path");
  expandIcon.id = `${id}-expand-icon`;
  expandIcon.setAttribute("d", "M5 5h10v10H5z");

  // Create the collapse icon element
  const collapseIcon = document.createElement("path");
  collapseIcon.id = `${id}-collapse-icon`;
  collapseIcon.setAttribute("d", "M7 10h6");

  // Add the icons to the toggle button element
  toggleBtn.appendChild(expandIcon);
  toggleBtn.appendChild(collapseIcon);

  // Create the content element
  const content = document.createElement("div");
  content.id = `${id}-box-content`;
  content.classList.add("mt-4");
  content.textContent = text;
  // Add the title, toggle button, and content to the box element
  // Create the title and toggle button container element
  const titleToggleContainer = document.createElement("div");
  titleToggleContainer.classList.add("flex", "justify-between", "items-center");

  // Add the title and toggle button to the container element
  titleToggleContainer.appendChild(title);
  titleToggleContainer.appendChild(toggleBtn);

  // Add the container element to the box element
  box.appendChild(titleToggleContainer);
  box.appendChild(content);

  // Add the box to the document
  document.body.appendChild(box);

  // Add the CSS styles to the head of the document
  const css = `
      #${id}-box-content {
        max-height: 0;
        overflow: hidden;
        transition: max-height 0.2s ease-out;
      }
  
      #${id}-box-content.expanded {
        max-height: 1000px;
        transition: max-height 0.5s ease-in;
      }
  
      #${id}-toggle-btn:focus #${id}-collapse-icon {
        display: block;
      }
  
      #${id}-toggle-btn:focus #${id}-expand-icon {
        display: none;
      }
  
      #${id}-collapse-icon {
        display: none;
      }
    `;
  const head = document.head || document.getElementsByTagName("head")[0];
  const style = document.createElement("style");
  style.type = "text/css";
  style.appendChild(document.createTextNode(css));
  head.appendChild(style);

  // Add the JavaScript code to toggle the box
  const toggleBtnEl = document.querySelector(`#${id}-toggle-btn`);
  const boxContentEl = document.querySelector(`#${id}-box-content`);

  toggleBtnEl.addEventListener("click", function () {
    boxContentEl.classList.toggle("expanded");
  });
  return box;
}

const welcome_message = `
<div>
<code>这是ChatCAD-plus的早期测试版本，欢迎任何反馈和贡献
</code>
<code>
<br>
- 将胸片、牙片等图像上传至聊天框，即可获得ChatCAD-plus对该影像的分析<br>
- 可以继续与ChatCAD-plus交流，进一步了解可能的病症<br>
- ChatCAD-plus会在必要的时候给出相关资料的链接<br>
</code>
</div>
`;
//welcome_message = add_collapsible_div("Note:", text, 'hints');

addMessage("ChatCAD-plus", welcome_message, 0);

// Code for collapsable text
const collapsibles = document.querySelectorAll(".collapsible");
function uncollapse(id) {
  console.log("uncollapsing");
  const content = document.querySelector(`#${id}`);
  content.classList.toggle("active");
}
