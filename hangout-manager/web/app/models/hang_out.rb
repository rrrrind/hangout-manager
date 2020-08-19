class HangOut < ApplicationRecord
  belongs_to :user
  has_many :likes
  has_many :liked_users, through: :likes, source: :user
  has_many :comments, dependent: :destroy

  enum label: { 募集中: 0, 募集締め切り: 1, 終了: 2 }
end
