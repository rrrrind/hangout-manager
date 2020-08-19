Rails.application.routes.draw do
  resources :recommend_users
  get 'users/:id/show', :to => "users#show"
  resources :questions
  # 最初の画面
  root 'welcome#welcome'

  # hang_out系
  resources :hang_outs do
    resources :likes, only: [:create, :destroy]
    resources :comments, only: [:create]
  end
  get "hang_outs/:id/json", :to => "hang_outs#get_json"
  get "hang_outs/:id/confirm", :to => "hang_outs#confirm_like"
  get "hang_outs/:id/confirm_destroy", :to => "hang_outs#confirm_destroy"

  # user_detail 系
  resources :user_details
  get "user_details/:id/json", :to => "user_details#get_json"
  
  # user_personal 系
  resources :user_personals
  get "user_personals/:id/json", :to => "user_personals#get_json"
  get "user_all/json", :to => "user_personals#get_all_json"

  # recommend 系
  get "recommend_one/:id", :to => "hang_outs#recommend_one"
  get "recommend_two/:id", :to => "hang_outs#recommend_two"
  get "recommend_three/:id", :to => "hang_outs#recommend_three"
  get "recommend_four/:id", :to => "hang_outs#recommend_four"
  get "recommend_five/:id", :to => "hang_outs#recommend_five"
  get "recommend_create", :to => "hang_outs#recommend_create", as: 'recommend_create'
  
  # フォロー系
  resources :relationships, only: [:create, :destroy]

  # Question
  get "questions/:id/json", :to => "questions#get_json"

  get "to_current", :to => "recommend_users#to_current"
  get "to_current/:id/json", :to => "recommend_users#to_current_json"
  get "get_current/:id", :to => "recommend_users#get_current"

  # マイページ系
  get 'mypage/show', as: 'mypage'
  # For details on the DSL available within this file, see https://guides.rubyonrails.org/routing.html
  
  # devise系
  devise_for :users, :controllers => {
    :registrations => 'users/registrations',
    :sessions => 'users/sessions'   
  } 

  # ログイン系
  devise_scope :user do
    get "user/:id", :to => "mypage#show"
    get "user/:id/follow", :to => "mypage#follow"
    get "user/:id/follower", :to => "mypage#follower"
    get "signup", :to => "users/registrations#new"
    get "login", :to => "users/sessions#new"
    get "logout", :to => "users/sessions#destroy"
  end 
end
